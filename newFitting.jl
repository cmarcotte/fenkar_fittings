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
		
		#		tsi,  tv1m,  tv2m,  tvp,  twm,  twp,  td,  to,  tr,  xk,  uc,  uv,  ucsi
		#=
		#P[1:13] .= [	30.0, 1250.0, 19.6, 3.33, 41.0, 870.0, 0.25, 12.5, 33.0, 10.0, 0.13, 0.04, 0.85] 	# BR
		#P[1:13] .= [	22.0, 333.0, 40.0, 10.0, 65.0, 1000.0, 0.12, 12.5, 25.0, 10.0, 0.13, 0.025, 0.85] 	# GP
		#P[1:13] .= [	127.0, 38.2, 38.2, 1.62, 80.0, 1020.0, 0.1724, 12.5, 130.0, 10.0, 0.13, 0.05, 0.85]	# barone
		#P[1:13] .= [	146.0, 373.0, 124.0, 16.3, 682.0, 529.0, 0.156, 18.0, 81.6, 10.5, 0.339, 0.293, 0.691]	# fit
		#P[1:13] .= [	146.0, 15.0, 15.0, 16.3, 200.0, 100.0, 0.156, 10.0, 50.0, 10.5, 0.339, 0.293, 0.691]	# mf35
		fitting.pinModelParams!(P, lb, ub; tol=0.0);
		=#
		
		result, l, sol = optimizeParams(P, lb, ub; u0=u0, tspan=tspan, t=t, Nt=Nt, Nsols=Nsols, target=target, M=M);
		
		if l < ltol
			push!(LL, l);
			push!(PP, result.u[1:13]);
			writeParams(result.u, l, "./fittings/Nt_$(Nt)/all_params.txt");
			cp("./fittings/Nt_$(Nt)/all_params.txt", "./fittings/Nt_$(Nt)/$(length(PP)).txt");
		end

	end
	#ll(x) = loss(x; u0=u0, tspan=tspan, t=t, Nt=Nt, Nsols=Nsols, target=target, M=(x)->(1.0));
	#ls, sols = reviseFits(Nt, ll);
	#plotAllFits(t, ls, sols; Nt=Nt, Nsols=Nsols, target=target);
	distros(Float64.(reduce(hcat,PP)),LL,Nt);
	paramCovariance(Float64.(reduce(hcat,PP)),LL,Nt);
	return nothing
end

function main()
	appendFitParams();
	return nothing
end

main()
