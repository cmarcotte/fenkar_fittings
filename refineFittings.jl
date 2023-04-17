const Nt = 500

include("dataManagement.jl"); include("plotting.jl"); include("fitting.jl")
using .dataManagement, .plotting, .fitting
using Optimization, OptimizationNLopt

const dt = 2.0				# time between samples -- set by the data
const u0 = rand(Float64,3)
const Nsols = 62
const tspan = (0.0,Nt*dt)
const t = collect(range(tspan[1], tspan[2]; length=Nt));
const target = zeros(Float64, 1, Nt, Nsols);
const t0s = zeros(Float64, Nsols);
const BCLs = zeros(Float64, Nsols);
loadTargetData!(target, t0s, BCLs; Nt=Nt)
const LL, PP = knownFits(Nt; truncate=false);

function pinExceptSetParams!(P, lb, ub, ns::T, n) where T <: Integer
	m = 13 .+ (ns-1)*5 .+ (1:5);
	for mm in 14:length(P)
		if !(mm in m)
			P[ns]  = PP[n][ns];
			lb[ns] = PP[n][ns];
			ub[ns] = PP[n][ns];
		end
	end
	return nothing
end

function pinExceptSetParams!(P, lb, ub, inds::AbstractArray{T}, n) where T <: Integer
	for ns in inds
		pinExceptSetParams!(P, lb, ub, ns, n)	
	end
	return nothing
end

function refineFitParams(n)
	
		P, lb, ub = parametersAndBounds(Nt, Nsols, t0s, BCLs);
		P[:] .= PP[n][:];
		checkBoundValidity!(P, ub, lb);
		fitting.pinModelParams!(P, ub, lb; tol=0.0);
		
		ll(x,p) = loss(x; u0=u0, tspan=tspan, t=t, Nt=Nt, Nsols=Nsols, target=target, M=(x)->1.0)	
		f = OptimizationFunction(ll, Optimization.AutoForwardDiff())
		l0, sol0 = f(PP[n], nothing);
		
		global iter = Int(0);
		cb = function (θ,l,sol) # callback function to observe training
			global iter = iter + 1;
			if isinf(l) || isnothing(l)
				return true
			elseif iter == 0 || iter == 1 || mod(iter,100) == 0
				print("Iter = $(iter), \tl = $(round(l;digits=fitting.sigdigs)), \tl/l0 = $(round(l/l0;digits=fitting.sigdigs)).\n");
			end
			return false
		end
				
		# Multi-stage Optimization:
		#	Global:
		optProb = OptimizationProblem(f, P, p = SciMLBase.NullParameters(), lb=lb, ub=ub);
		result = solve(optProb, NLopt.GN_MLSL_LDS(), local_method = NLopt.LN_COBYLA(); callback = cb, maxiters=10_000, abstol=sqrt(fitting.atol), reltol=sqrt(fitting.rtol), xtol_rel=sqrt(fitting.atol), xtol_abs=sqrt(fitting.rtol));
		#	Local:
		optProb = OptimizationProblem(f, result.u, p = SciMLBase.NullParameters(), lb=lb, ub=ub);
		result = solve(optProb, NLopt.LD_SLSQP(); callback = cb, abstol=fitting.atol, reltol=fitting.rtol, xtol_rel=fitting.atol, xtol_abs=fitting.rtol);
		
		l, sol = ll(result.u, nothing);
		
		print("="^80)
		print("\n\tOriginal loss is $(l0), new loss is $(l).\n")
		print("Model params preserved? $(all(result.u[1:13] .≈ P[1:13])).\n");
		print("Loss decreased? $(l < l0).\n");
		if l < l0 && all(result.u[1:13] .≈ P[1:13])
			writeParams(result.u, l, "./fittings/Nt_$(Nt)/$(n).txt");
			LL[n] = l;
			PP[n][14:end] .= result.u[14:end];
			print("\tn=$(n) is updated.\n")
		else
			print("\tn=$(n) is not updated.\n")
		end
		print("="^80*"\n")

	return nothing
end

function main()
	Threads.@threads for n in 986:length(PP)
		refineFitParams(n);
	end
	return nothing
end

main()
