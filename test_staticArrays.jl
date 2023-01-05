using OrdinaryDiffEq, ForwardDiff, Optimization, OptimizationNLopt, StaticArrays, DelimitedFiles

# model stuff
function H(x;k=100.0)
	return 0.5*(1.0 + tanh(k*x))
end
function Monophasic(t,IA,t0,TI)
	return IA*H(t-t0;k=1.0)*sin(pi*(t-t0)/TI)^500
end	
function Biphasic(t,IA,t0,TI)
	return IA*H(t-t0;k=1.0)*500*(pi/TI)*cos(pi*(t-t0)/TI)*sin(pi*(t-t0)/TI)^499
end
Istim(t,IA,t0,TI) = Monophasic(t,IA,t0,TI);
function fenkar(x, p, t)
	
	# parameters
	@views tsi,tv1m,tv2m,tvp,twm,twp,td,to,tr,xk,uc,uv,ucsi,t0,TI,IA = p[1:16]
	
	# fenkar dynamics
	dx1 = Istim(t,IA,t0,TI) - (x[1]*H(uc-x[1])/to + H(x[1]-uc)/tr - x[2]*H(x[1]-uc)*(1.0-x[1])*(x[1]-uc)/td - x[3]*H(x[1]-ucsi;k=xk)/tsi)
	dx2 = H(uc-x[1])*(1.0-x[2])/(tv1m*H(uv-x[1]) + tv2m*H(x[1]-uv)) - H(x[1]-uc)*x[2]/tvp
	dx3 = H(uc-x[1])*(1.0-x[3])/twm - H(x[1]-uc)*x[3]/twp
	
	return SA[dx1,dx2,dx3]
end
function fenkar!(dx, x, p, t)
	
	# parameters
	@views tsi,tv1m,tv2m,tvp,twm,twp,td,to,tr,xk,uc,uv,ucsi,t0,TI,IA = p[1:16]
	
	# fenkar dynamics
	dx[1] = Istim(t,IA,t0,TI) - (x[1]*H(uc-x[1])/to + H(x[1]-uc)/tr - x[2]*H(x[1]-uc)*(1.0-x[1])*(x[1]-uc)/td - x[3]*H(x[1]-ucsi;k=xk)/tsi)
	dx[2] = H(uc-x[1])*(1.0-x[2])/(tv1m*H(uv-x[1]) + tv2m*H(x[1]-uv)) - H(x[1]-uc)*x[2]/tvp
	dx[3] = H(uc-x[1])*(1.0-x[3])/twm - H(x[1]-uc)*x[3]/twp
	
	return nothing
end


# loading options
const modelfile	= true
const stimfile	= true
const sigdigs	= 10
const skipPars  = 0

# make the parameters for the solution
const Nt = 1200
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
	M(P) = 1.0; #prod([a + 1.0./sum(abs2,P.-p) for p in PP])
	return M
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
	end
	if stimfile
		P[14:length(P)] .= transpose(readdlm("./fittings/Nt_$(Nt)/stim_params.txt"))[:];
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
	prob = ODEProblem(fenkar, SA[u0...], tspan, SA[θ[1:16]...])
	
	function prob_func(prob, i, repeat)
		remake(prob, u0 = SA[target[1,1,i], θ[17+5*(i-1)], θ[18+5*(i-1)]], p = SA[θ[[1:13; (14+5*(i-1)):(16+5*(i-1))]]...])
	end

	ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
	sim = solve(ensemble_prob, Tsit5(), ensemble, saveat = t, save_idxs=1:1, trajectories = 62, maxiters=Int(1e8), abstol=1e-8, reltol=1e-6)
	
end

# model definition using (ensemble) ODE problem
function model2(θ,ensemble)
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
function loss1(θ, _p; ensemble=EnsembleThreads())
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

# define loss function
function loss2(θ, _p; ensemble=EnsembleThreads())
	sol = model2(θ,ensemble)
	if any((s.retcode != :Success for s in sol))
		l = Inf
	elseif size(Array(sol)) != size(target[:,:,1:Nsols])
		print("I'm a doodoohead poopface") 
	else
		l = sum(abs2, (target[:,:,1:Nsols].-Array(sol))) * M(θ[1:13])/(Nt*Nsols)
	end
	return l,sol
end

@benchmark sol1 = model1(P, EnsembleSerial())
#=
BenchmarkTools.Trial: 41 samples with 1 evaluation.
 Range (min … max):  116.912 ms … 131.804 ms  ┊ GC (min … max): 0.00% … 8.08%
 Time  (median):     126.029 ms               ┊ GC (median):    6.02%
 Time  (mean ± σ):   124.394 ms ±   4.785 ms  ┊ GC (mean ± σ):  4.77% ± 3.49%

    ▄▁▄▁                               █  ▁  ▄ ▄   ▁▁    ▁       
  ▆▆████▁▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▆▁▁▆▁▁▆▁▆▆█▁▆█▁▆█▆█▁▆▁██▁▁▁▁█▁▁▁▁▆ ▁
  117 ms           Histogram: frequency by time          132 ms <

 Memory estimate: 107.72 MiB, allocs estimate: 1712301.
=#
@benchmark sol2 = model2(P, EnsembleSerial())
#=
BenchmarkTools.Trial: 65 samples with 1 evaluation.
 Range (min … max):  76.187 ms … 85.079 ms  ┊ GC (min … max): 0.00% … 9.40%
 Time  (median):     77.215 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   77.568 ms ±  1.686 ms  ┊ GC (mean ± σ):  0.44% ± 1.85%

  ▂ ▃  █▂ ▂▂  ▃                                                
  █▄██▅██████▅█▅▄▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▄ ▁
  76.2 ms         Histogram: frequency by time        84.8 ms <

 Memory estimate: 8.07 MiB, allocs estimate: 79065.
=#
@benchmark l1,sol1 = loss1(P,SciMLBase.NullParameters())
#=
BenchmarkTools.Trial: 113 samples with 1 evaluation.
 Range (min … max):  35.446 ms … 74.495 ms  ┊ GC (min … max):  0.00% … 47.59%
 Time  (median):     38.276 ms              ┊ GC (median):     0.00%
 Time  (mean ± σ):   44.670 ms ± 14.276 ms  ┊ GC (mean ± σ):  15.79% ± 19.29%

   █▂ ▁▅                                                       
  ███▅██▄▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▃▅▅▅▄ ▃
  35.4 ms         Histogram: frequency by time        74.2 ms <

 Memory estimate: 112.86 MiB, allocs estimate: 1712732.
=#
@benchmark l2,sol2 = loss2(P,SciMLBase.NullParameters())
#=
BenchmarkTools.Trial: 196 samples with 1 evaluation.
 Range (min … max):  24.279 ms … 40.808 ms  ┊ GC (min … max): 0.00% … 37.59%
 Time  (median):     24.924 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   25.606 ms ±  2.917 ms  ┊ GC (mean ± σ):  2.29% ±  7.19%

  ▃█▄▅▅▃                                                       
  ██████▆▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▆▁▄▁▆ ▄
  24.3 ms      Histogram: log(frequency) by time        40 ms <

 Memory estimate: 13.16 MiB, allocs estimate: 79421.
=#