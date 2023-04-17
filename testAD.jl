const Nt = 500
const N = 10

include("dataManagement.jl"); include("plotting.jl"); include("fitting.jl")
using .dataManagement, .plotting, .fitting

const dt = 2.0				# time between samples -- set by the data
const u0 = rand(Float64,3)
const Nsols = 62
const tspan = (0.0,Nt*dt)
const t = collect(range(tspan[1], tspan[2]; length=Nt));
const target = zeros(Float64, 1, Nt, Nsols);
const t0s = zeros(Float64, Nsols);
const BCLs = zeros(Float64, Nsols);
loadTargetData!(target, t0s, BCLs; Nt=Nt)
const LL, PP = knownFits(Nt; truncate=true);

using Optimization, OptimizationNLopt, ForwardDiff, ReverseDiff, Zygote

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

cb = function (θ,l,sol) # callback function to observe training
    	global iter = iter + 1;
	if mod(iter,1) == 0 || iter == 0
		print("Iter = $(iter), \tLoss = $(round(l;digits=fitting.sigdigs)).\n");
	end
	if isinf(l) || isnothing(l)
		return true
	end
    	return false
end

function compareAD()

	results = [Dict() for n in 1:N]
	ADs = [Optimization.AutoForwardDiff(), Optimization.AutoReverseDiff(compile=false), Optimization.AutoZygote()]
	for n in 1:N
		M = deflationOperator(PP);
		
		P, lb, ub = parametersAndBounds(Nt, Nsols, t0s, BCLs);
		
		ll(x,p) = loss(x; u0=u0, tspan=tspan, t=t, Nt=Nt, Nsols=Nsols, target=target, M=M)
	
		# initial loss
		l1,sol1 = ll(P, nothing);

		for AD in ADs
			
			# Optimization function:
			f = OptimizationFunction(ll, AD)
			
			# Optimization Problem definition:
			optProb = OptimizationProblem(f, P, p = SciMLBase.NullParameters(), lb=lb, ub=ub);
			
			# reset iter counter
			global iter = Int(0)
			
			# Optimization:
			print("\nOptimizing with NLopt.LD_SLSQP() and $AD:\n")
			result = solve(optProb, NLopt.LD_SLSQP(); callback=cb, abstol=1e-8, reltol=1e-6, xtol_rel=1e-8, xtol_abs=1e-6)
			results[n]["$AD"] = result
		end
	end
	
	return results
end

function main()
	results = compareAD();
	
	
	
	return nothing
end

main()
