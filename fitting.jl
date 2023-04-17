__precompile__()
module fitting
include("model.jl")
using .fenkar, OrdinaryDiffEq, Random, ForwardDiff, Optimization, OptimizationNLopt, DelimitedFiles

export cb, deflationOperator, model, loss, optimizeParams, parametersAndBounds, checkBoundValidity!
export ltol

const sigdigs = 10
const ltol = 0.15^2
const atol = 1e-8
const rtol = 1e-6

function pinModelParams!(P,ub,lb; tol=0.0)
	inds = 1:13;
	ub[inds] .= P[inds].*(1.0 .+ sign.(P[inds]).*tol)
	lb[inds] .= P[inds].*(1.0 .- sign.(P[inds]).*tol)
	return nothing
end

function pinStimParams!(P,ub,lb; tol=0.0)
	inds = 14:length(P);
	ub[inds] .= P[inds].*(1.0 .+ sign.(P[inds]).*tol)
	lb[inds] .= P[inds].*(1.0 .- sign.(P[inds]).*tol)
	return nothing
end
function checkBoundValidity!(P, ub, lb; tol=1.0)
	# and then check lb/ub are valid
	li = findall(P .<= lb)
	if !isempty(li)
		@debug "$(lb[li]) .> $(P[li])"
		lb[li] .= P[li].*(1.0 .- sign.(P[li]).*tol)
	end
	ui = findall(P .>= ub)
	if ~isempty(ui)
		@debug "$(ub[ui]) .< $(P[ui])"
		ub[ui] .= P[ui].*(1.0 .+ sign.(P[ui]).*tol)
	end
	return nothing
end
function parametersAndBounds(Nt, Nsols, t0s, BCLs; modelfile=false, stimfile=false)
	# set up model parameters
	P = zeros(Float64, 13 + 5*Nsols)
	lb= zeros(Float64, 13 + 5*Nsols)
	ub= zeros(Float64, 13 + 5*Nsols)
	#####		tsi,	tv1m,	tv2m,	tvp,	twm,	twp,	td,	to,	tr,	xk,	uc,	uv,	ucsi
	lb[1:13].= [ 	1.0, 	1.0, 	1.0, 	1.0, 	1.0, 	1.0, 	0.05, 	1.0, 	1.0, 	9.0, 	0.10, 	0.01, 	0.10 	][1:13]
	ub[1:13].= [ 	1000.0, 1000.0, 2000.0, 15.0,	1000.0,	1000.0,	0.50, 	1000.0,	1000.0,	11.0,	0.15,	0.05,	0.90 	][1:13]
	#P[1:13] .= [	29.0	19.6	1250.0	3.33	41.0	870.0	0.25	12.5	33.3	10.0	0.13	0.04	0.85	][1:13]
	for n in 1:13
		P[n] = rand(range(lb[n], ub[n]; length=10^6));
	end
	
	# initialize stimulus parameters with randomly chosen values, adapt lb/ub to fit
	for m in 1:Nsols
		P[13 + 5*(m-1) + 1] = t0s[m]					# stimlus offset / phase
		P[13 + 5*(m-1) + 2] = BCLs[m]					# stimulus period
		P[13 + 5*(m-1) + 3] = min(max(0.0,0.159 + 0.025*randn()),0.25)	# stimulus amplitude
		P[13 + 5*(m-1) + 4] = rand()					# v(0) ~ U[0,1] 
		P[13 + 5*(m-1) + 5] = rand()					# w(0) ~ U[0,1]
		
		lb[13 + 5*(m-1) + 1] = t0s[m]-min(BCLs[m]/2.0,10.0)
		lb[13 + 5*(m-1) + 2] = BCLs[m]-min(BCLs[m]*0.1,10.0)
		lb[13 + 5*(m-1) + 3] = 0.0
		lb[13 + 5*(m-1) + 4] = 0.0
		lb[13 + 5*(m-1) + 5] = 0.0
		
		ub[13 + 5*(m-1) + 1] = t0s[m]+min(BCLs[m]/2.0,10.0)
		ub[13 + 5*(m-1) + 2] = BCLs[m]+min(BCLs[m]*0.1,10.0)
		ub[13 + 5*(m-1) + 3] = 0.25
		ub[13 + 5*(m-1) + 4] = 1.0
		ub[13 + 5*(m-1) + 5] = 1.0
	end
	if modelfile
		# and write over from the file(s) if using those
		P[1:13] .= transpose(readdlm("./fittings/Nt_$(Nt)/model_params.txt"))[1:13];
		if !stimfile
			pinModelParams!(P,ub,lb); # can't freeze both!
		end
	end
	if stimfile
		P[14:length(P)] .= transpose(readdlm("./fittings/Nt_$(Nt)/stim_params.txt"))[:];
		if !modelfile
			pinStimParams!(P,ub,lb); # can't freeze both!
		end
	end
	checkBoundValidity!(P, ub, lb);
	
	return P, lb, ub
end

function deflationOperator(PP; a=1.0)
	M(P) = prod([a + 1.0./sum(abs2,P.-p) for p in PP])
	return M
end

iter = Int(0)
cb = function (θ,l,sol; plotting=false) # callback function to observe training
    global iter = iter + 1;
	if isinf(l) || isnothing(l)
		return true
	elseif mod(iter,10) == 0
		print("Iter = $(iter), \tLoss = $(round(l;digits=sigdigs)), \tSQRT Loss=$(round(sqrt(l);digits=sigdigs)).\n");
		if sqrt(l) < 0.15 && mod(iter, 100) == 0
			saveProgress(iter,θ,l,sol; plotting=plotting);
		end
	end
    return false
end

# model definition using (ensemble) ODE problem
function model(θ; u0=rand(Float64,3), tspan=(0.0,1.0), t=[], Nt=1, Nsols=1, target=rand(Float64,1,1,1))
	
	# base ODE problem
	prob = ODEProblem(fenkar!, u0, tspan, θ[1:16]);
	
	# ensemble initialization of parameters for problems
	function prob_func(prob, i, repeat)
		remake(prob, u0 = [target[1,1,i], θ[17+5*(i-1)], θ[18+5*(i-1)]], p = θ[[1:13; (14+5*(i-1)):(16+5*(i-1))]])
	end

	ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
	sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), saveat = t, save_idxs=1:1, trajectories = Nsols, maxiters=Int(1e8), abstol=atol, reltol=rtol)
	return sim
end

# define loss function
function loss(θ; u0=rand(Float64,3), tspan=(0.0,1.0), t=[], Nt=1, Nsols=1, target=rand(Float64,1,1,1), M=nothing)
	sol = fitting.model(θ; u0=u0, tspan=tspan, t=t, Nt=Nt, Nsols=Nsols, target=target)
	#if any((s.retcode != :Success for s in sol))
	if any([!SciMLBase.successful_retcode(s) for s in sol])
		l = Inf
		print("Inf loss! Error in test?")
	elseif size(Array(sol)) != size(target[:,:,1:Nsols])
		print("Solution does not match target size!") 
	else
		l = sum(abs2, (target[:,:,1:Nsols].-Array(sol))) * M(θ[1:13])/(Nt*Nsols)
	end
	return l,sol
end

function optimizeParams(P, lb, ub; u0=rand(Float64,3), tspan=(0.0,1.0), t=[], Nt=1, Nsols=1, target=rand(Float64,1,1,1), M=nothing, AD=Optimization.AutoForwardDiff(), method=NLopt.LD_SLSQP())
	
	ll(x,p) = loss(x; u0=u0, tspan=tspan, t=t, Nt=Nt, Nsols=Nsols, target=target, M=M)
	
	# initial loss
	l1,sol1 = ll(P, nothing);
	
	# reset iter counter
	global iter = Int(0)
	
	if AD != Optimization.AutoForwardDiff()
		@warn "Testing of alternating AD schemes indicates they don't work at all!"
	end
	# Optimization function:
	f = OptimizationFunction(ll, AD)
	
	# Optimization Problem definition:
	optProb = OptimizationProblem(f, P, p = SciMLBase.NullParameters(), lb=lb, ub=ub);
	
	# Optimization:
	print("\nOptimizing with $method and $AD:\n")
	result = solve(optProb, method; callback=cb, abstol=atol, reltol=rtol, xtol_rel=atol, xtol_abs=rtol, maxiters=5000)
	
	if result.objective >= ltol
		
		# Pin the model parameters to their optimal values:
		pinModelParams!(result.u, ub, lb; tol=0.0);
		optProb = OptimizationProblem(f, result.u, p = SciMLBase.NullParameters(), lb=lb, ub=ub);
		
		# Optimization:
		print("\nOptimizing with NLopt.GN_MLSL_LDS() with NLopt.LN_SBPLX():\n")
		#result = solve(optProb, NLopt.GN_DIRECT_L(); callback=cb, abstol=atol, reltol=rtol, xtol_rel=atol, xtol_abs=rtol, maxiters=50000-iter)
		result = solve(optProb, NLopt.GN_MLSL_LDS(), local_method = NLopt.LN_SBPLX(); callback=cb, abstol=atol, reltol=rtol, xtol_rel=atol, xtol_abs=rtol, maxiters=50000-iter)
	end
	
	l,sol = ll(result.u, nothing);
	print("\n\tFinal loss: $(l); Initial loss: $(l1).\n")

	return result, l, sol
	
end

end
