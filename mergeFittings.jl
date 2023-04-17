const Nt = 1200

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

function refineFitParams(Q)
		
	# load known fits for Nt
	LL, PP = knownFits(Nt; truncate=true);
	PP = [Float64.(P) for P in PP];
	MM = deflationOperator(PP; a=1.0);
	
	# 
	n = length(PP) + 1;
	@assert length(LL) + 1 == n
	
	# form P and bounds for input (fit from Mt)
	P, lb, ub = parametersAndBounds(Nt, Nsols, t0s, BCLs);
	P[:] .= Q[:];
	checkBoundValidity!(P, ub, lb);
	fitting.pinModelParams!(P, ub, lb; tol=0.0);
	
	if !isinf(MM(P[1:13])) && !isnan(MM(P[1:13]))
		
		# form the loss function based on PP -- with deflation!
		lss(x,p) = loss(x; u0=u0, tspan=tspan, t=t, Nt=Nt, Nsols=Nsols, target=target, M=(x)->MM(x[1:13]))	
		f = OptimizationFunction(lss, Optimization.AutoForwardDiff())

		# evaluate it on the off-chance the input is too close to an existing fit for Nt
		l0, sol0 = f(P, nothing);

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
		# 	final:
		l, sol = lss(result.u, nothing);

		print("="^80)
		print("\n\tOriginal loss is $(l0), new loss is $(l).\n")
		print("Model params preserved? $(all(result.u[1:13] .≈ P[1:13])).\n");
		print("Loss decreased? $(l < l0).\n");
		if l < l0 && all(result.u[1:13] .≈ P[1:13])
			writeParams(result.u, l, "./fittings/Nt_$(Nt)/$(n).txt");
			print("\tn=$(n) is updated.\n")
		elseif l0 < ltol			
			writeParams(P, l0, "./fittings/Nt_$(Nt)/$(n).txt");
			print("\tn=$(n) is appended.\n")
		else
			print("\tn=$(n) is not updated.\n")
		end
		print("="^80*"\n")
	else
		print("\n Skipping $(n)...\n");
		print("="^80*"\n")
	end

	return nothing
end

function main()
	for Mt in [250, 500, 1000, 1200]
		if Nt != Mt
			ll,pp = knownFits(Mt; truncate=false)
			for p in pp
				refineFitParams(p);
			end
		end
	end
	return nothing
end

main()
