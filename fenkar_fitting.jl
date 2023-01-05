const Nt = parse(Int, ARGS[1])

include("./model.jl")
using .fenkar
using OrdinaryDiffEq
using ForwardDiff, Optimization, OptimizationNLopt
using Random, DelimitedFiles

using PyPlot
plt.style.use("seaborn-paper")
PyPlot.rc("font", family="serif")
PyPlot.rc("text", usetex=true)
PyPlot.matplotlib.rcParams["axes.titlesize"] = 10
PyPlot.matplotlib.rcParams["axes.labelsize"] = 10
PyPlot.matplotlib.rcParams["xtick.labelsize"] = 9
PyPlot.matplotlib.rcParams["ytick.labelsize"] = 9
const sw = 3.40457
const dw = 7.05826

# test for directories, and if not there, make them
function init()
	if !isdir("./data")
		mkdir("./data/");
	end
	if !isdir("./fittings")
		mkdir("./fittings/");
	end
	return nothing
end
init();

# loading options
const modelfile	= true
const stimfile	= true
const sigdigs	= 10
const skipPars  = 0

# make the parameters for the solution
const dt = 2.0			# time between samples -- set by the data
const tspan = (0.0,Nt*dt)
const t = collect(range(tspan[1], tspan[2]; length=Nt));
const u0 = rand(Float64,3)

const Nsols = 62
const target = zeros(Float64, 1, Nt, Nsols);
const t0s = zeros(Float64, Nsols);
const BCLs = zeros(Float64, Nsols);
try
	target .= reshape(readdlm("./data/Nt_$(Nt)/target.txt"), size(target));
	t0s .= reshape(readdlm("./data/Nt_$(Nt)/t0s.txt"), size(t0s));
	BCLs .= reshape(readdlm("./data/Nt_$(Nt)/BCLs.txt"), size(BCLs));
catch
	print("uh oh, spaghettios!\n")
end

function knownParameters()
	PP = []
	loading = true
	ind = 0
	while loading
		try
			ind = ind + 1
			tmp = readdlm("./fittings/Nt_$(Nt)/$(ind).txt"; comments=true, comment_char='#');
			push!(PP, tmp[1,1:13][:])
		catch
			loading = false
		end
	end
	print("There are $(length(PP)) known model parameter sets!\n");
	return PP
end

function deflationOperator(PP; a=1.0)
	M(P) = prod([a + 1.0./sum(abs2,P.-p) for p in PP])
	return M
end

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

function parametersAndBounds()
	# set up model parameters
	P = zeros(Float64, 13 + 5*Nsols)
	lb= zeros(Float64, 13 + 5*Nsols)
	ub= zeros(Float64, 13 + 5*Nsols)
	#####			tsi,	tv1m,	tv2m,	tvp,	twm,	twp,	td,		to,		tr,		xk,		uc,		uv,		ucsi
	P[1:13] .= [	29.0	19.6	1250.0	3.33	41.0	870.0	0.25	12.5	33.3	10.0	0.13	0.04	0.85	][1:13]
	lb[1:13].= [ 	1.0, 	1.0, 	1.0, 	1.0, 	1.0, 	1.0, 	0.05, 	1.0, 	1.0, 	9.0, 	0.10, 	0.01, 	0.10 	][1:13]
	ub[1:13].= [ 	1000.0, 1000.0, 2000.0, 15.0,	1000.0,	1000.0,	0.50, 	1000.0,	1000.0,	11.0,	0.15,	0.05,	0.90 	][1:13]
	
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
		pinModelParams!(P,ub,lb);
	end
	if stimfile
		P[14:length(P)] .= transpose(readdlm("./fittings/Nt_$(Nt)/stim_params.txt"))[:];
		if !modelfile
			pinStimParams!(P,ub,lb);
		end
	end
	# and then check lb/ub are valid
	li = findall(P .<= lb)
	if !isempty(li)
		@show "$(lb[li]) .> $(P[li])"
		lb[li] .= P[li].*0.5
	end
	ui = findall(P .>= ub)
	if ~isempty(ui)
		@show "$(ub[ui]) .< $(P[ui])"
		ub[ui] .= P[ui].*2.0
	end
	
	return P, lb, ub
end



P, lb, ub = parametersAndBounds();

# model definition using (ensemble) ODE problem
function model1(θ,ensemble)
	prob = ODEProblem(fenkar!, u0, tspan, θ[1:16])
	
	function prob_func(prob, i, repeat)
		remake(prob, u0 = [target[1,1,i], θ[17+5*(i-1)], θ[18+5*(i-1)]], p = θ[[1:13; (14+5*(i-1)):(16+5*(i-1))]])
	end

	ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
	sim = solve(ensemble_prob, Tsit5(), ensemble, saveat = t, save_idxs=1:1, trajectories = 62, maxiters=Int(1e8), abstol=1e-8, reltol=1e-6)
	
end

# get known parameters and form deflation operator
const PP = knownParameters();
M = deflationOperator(PP);
if isinteger(skipPars) && 1 <= skipPars && skipPars <= length(PP) && length(PP) > 0
	QQ = view(PP, vcat(1:(skipPars-1),(skipPars+1):length(PP)))
	M = deflationOperator(QQ);
end

# define loss function
function loss(θ, _p; ensemble=EnsembleThreads())
	sol = model1(θ,ensemble)
	if any((s.retcode != :Success for s in sol))
		l = Inf
	elseif size(Array(sol)) != size(target[:,:,1:Nsols])
		print("I'm a doodoohead poopface") 
	else
		l = sum(abs2, (target[:,:,1:Nsols].-Array(sol))) * M(θ[1:13])/(Nt*Nsols)
	end
	return l,sol
end

# initial loss and solution set
const l1,sol1 = loss(P,SciMLBase.NullParameters()); 
const l_tol = 0.15^2

# To get the Gradient or Hessian uncomment the function definition below:
# LOSS(P) = loss(P,SciMLBase.NullParameters())[1]
# A = ForwardDiff.hessian(LOSS,P);

function plotFits(θ,sol; target=target)

	fig, axs = plt.subplots(Int(ceil(Nsols/8)),8, figsize=(dw,dw*1.05*(Int(ceil(Nsols/8))/8)), 
				sharex=true, sharey=true, constrained_layout=true)
	xt = collect(0:4).*(Nt*dt/4);
	for n in 1:Nsols
		# linear indexing into Array{Axis,2}
		axs[n].plot(t, target[1,:,n], "-k", linewidth=1.6)
		# plotting the (negated) stimulus current to make sure things line up
		axs[n].plot(t, Istim.(t,-θ[13+5*(n-1)+3],θ[13+5*(n-1)+1],θ[13+5*(n-1)+2]), "-r", linewidth=0.5)
		axs[n].plot(t, sol1[1,:,n], "-C0", linewidth=1)
		axs[n].plot(t, sol[1,:,n], "-C1", linewidth=1)
	end
	axs[1].set_ylim([-0.1,1.1])
	axs[1].set_xlim([t[begin],t[end]])
	axs[1].set_xticks(xt)
	axs[1].set_xticklabels(["","$(round(xt[2]))","","$(round(xt[4]))",""])
	return fig, axs
end

function writeParams(θ, l, filename)
	open(filename, "w") do io
		write(io, "# Loss = $(l)\n\n")
		write(io, "# tsi\t\ttv1m\t\ttv2m\t\ttvp\t\ttwm\t\ttwp\t\ttd\t\tto\t\ttr\t\txk\t\tuc\t\tuv\t\tucsi\n")
		writedlm(io, transpose(round.(θ[1:13],sigdigits=sigdigs)))
		write(io,"\n")
		write(io, "# t0\t\tTI\t\tIA\t\tv0\t\tw0\n")
		writedlm(io, transpose(reshape(round.(θ[14:end],sigdigits=sigdigs-1), 5, :)))
	end
	#=
	open("./fittings/Nt_$(Nt)/model_params.txt", "w") do io
		writedlm(io, transpose(round.(θ[1:13],sigdigits=sigdigs)))
	end
	open("./fittings/Nt_$(Nt)/stim_params.txt", "w") do io
		writedlm(io, transpose(reshape(round.(θ[14:end],sigdigits=sigdigs-1), 5, :)))
	end
	=#
	return nothing;
end

function appendParams(P1, P2, filename)
	open(filename, "a") do io
		write(io, "\n")
		write(io, "# Loss (initial) = $(loss(P1,nothing)[1])\n")
		write(io, "# Loss (final) = $(loss(P2,nothing)[1])\n")
		Q = P2; Q[1:13] .= P[1:13];	
		write(io, "# Loss (resample) = $(loss(Q,nothing)[1])\n")
	end
	return nothing
end

function saveprogress(ind,θ,l,sol; plotting=false)
	writeParams(θ,l,"./fittings/Nt_$(Nt)/all_params.txt");
	if plotting
		fig, axs = plotFits(θ,sol)
		fig.savefig("./fittings/Nt_$(Nt)/all_fits.pdf",bbox_inches="tight")
		plt.close(fig)
	end
	return nothing
end

iter = Int(0)
cb = function (θ,l,sol; plotting=false) # callback function to observe training
    global iter = iter + 1;
	if isinf(l) || isnothing(l)
		return true
	elseif mod(iter,10) == 0
		print("Iter = $(iter), \tLoss = $(round(l;digits=sigdigs)), \tSQRT Loss=$(round(sqrt(l);digits=sigdigs)), \tReduction by $(round(100*(1-l/l1);digits=sigdigs))%.\n");
		if sqrt(l) < 0.15 && mod(iter, 100) == 0
			saveprogress(iter,θ,l,sol; plotting=plotting);
		end
	end
    return false
end

# Optimization function:
f = OptimizationFunction(loss, Optimization.AutoForwardDiff())

# Optimization Problem definition:
optProb = OptimizationProblem(f, P, p = SciMLBase.NullParameters(), lb=lb, ub=ub);

# Optimization:
print("\nOptimizing with NLopt.LD_SLSQP():\n")
result = solve(optProb, NLopt.LD_SLSQP(); callback=cb, abstol=1e-8, reltol=1e-6, xtol_rel=1e-8, xtol_abs=1e-6, maxiters=10000)

l,sol = loss(result.u, nothing);
print("\n\tFinal loss: $(l); Initial loss: $(l1).\n")

if l >= l_tol
	# Optimization function:
	f = OptimizationFunction(loss, Optimization.AutoForwardDiff())
	
	# Pin the model parameters to their optimal values:
	ub[1:13] .= result.u[1:13].*(1.0 .+ sign.(result.u[1:13]).*1e-6)
	lb[1:13] .= result.u[1:13].*(1.0 .- sign.(result.u[1:13]).*1e-6)
	optProb = OptimizationProblem(f, result.u, p = SciMLBase.NullParameters(), lb=lb, ub=ub);
	
	# Optimization:
	print("\nOptimizing with NLopt.GN_DIRECT_L():\n")
	result = solve(optProb, NLopt.GN_DIRECT_L(); callback=cb, abstol=1e-8, reltol=1e-6, xtol_rel=1e-8, xtol_abs=1e-6, maxiters=50000)
end

#=
	The loss is
		l = \sum_m=1^M \sum_n=1^N |u_m(t_n) - o_{mn}|^2
	so L = RMS error per sample is 
		L = sqrt(l/M/N)
	which we would like to be less than 10%, but because of the dimensionality of P we permit
	some wiggle room and set it to be < 15%.
=#

print("\n l = $(l), SQRT loss = $(sqrt(l)).\n");
if l < l_tol # 15% error threshold contribution to RMS per time-step per ode
	saveprogress(iter,result.u,l,sol; plotting=true)
	appendParams(P, result.u, "./fittings/Nt_$(Nt)/all_params.txt")
	cp("./fittings/Nt_$(Nt)/all_params.txt", "./fittings/Nt_$(Nt)/$(length(PP)+1).txt");
end
