const Nt = parse(Int, ARGS[1])
const N = parse(Int, ARGS[2])

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

function appendFitParams()

		
	for n in 1:N
		
		M = deflationOperator(PP);
		
		P, lb, ub = parametersAndBounds(Nt, Nsols, t0s, BCLs);
		
		result, l, sol = optimizeParams(P, lb, ub; u0=u0, tspan=tspan, t=t, Nt=Nt, Nsols=Nsols, target=target, M=M);
		
		if l < ltol
			push!(LL, l);
			push!(PP, result.u[1:13]);
			writeParams(result.u, l, "./fittings/Nt_$(Nt)/all_params.txt");
			cp("./fittings/Nt_$(Nt)/all_params.txt", "./fittings/Nt_$(Nt)/$(length(PP)).txt");
		end

	end
	ll(x) = loss(x; u0=u0, tspan=tspan, t=t, Nt=Nt, Nsols=Nsols, target=target, M=(x)->(1.0));
	ls, sols = reviseFits(Nt, ll);
	plotAllFits(t, ls, sols; Nt=Nt, Nsols=Nsols, target=target);
	distros(Float64.(reduce(hcat,PP)),LL,Nt);
	paramCovariance(Float64.(reduce(hcat,PP)),LL,Nt);
	return nothing
end

function main()
	appendFitParams();
	return nothing
end

main()