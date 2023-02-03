const Nt = 500

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

using Optimization, OptimizationNLopt

cb = function (θ,l,sol) # callback function to observe training
    global iter = iter + 1;
	if isinf(l) || isnothing(l)
		return true
	elseif mod(iter,1000) == 0
		print("Iter = $(iter), \tLoss = $(round(l;digits=sigdigs)).\n");
	end
    return false
end

function globalOptiz(;opts=[NLopt.GN_DIRECT_L(), NLopt.GN_CRS2_LM(), NLopt.GD_STOGO(), NLopt.GN_AGS(), NLopt.GN_ISRES(), NLopt.GN_ESCH()])
	
	P, lb, ub = parametersAndBounds(Nt, Nsols, t0s, BCLs);
	
	ll(x,p) = loss(x; u0=u0, tspan=tspan, t=t, Nt=Nt, Nsols=Nsols, target=target, M=(x)->1.0)
	
	P[1:13] .= PP[938][1:13];
	
	P[2] = 50.0
	P[3] = 80.0
	P[5] = 40.0
	P[6] = 300.0
	P[10] = 10.0; lb[10] = 10.0; ub[10] = 10.0;
	
	# Optimization function:
	f = OptimizationFunction(ll, Optimization.AutoForwardDiff())
	
	# Optimization Problem definition:
	optProb = OptimizationProblem(f, P, p = SciMLBase.NullParameters(), lb=lb, ub=ub);
	
	results = Dict()
	for optMethod in opts
		# reset iter counter
		global iter = Int(0)
		
		# Optimization:
		print("Optimizing with $optMethod:\n");
		result = solve(optProb, optMethod; callback = cb)
		results["$optMethod"] = result;
	end
	
	return results

end

function main()
	results = globalOptiz();
	return nothing
end

main()