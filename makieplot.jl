using CairoMakie
include("./model.jl")
using .fenkar
using DifferentialEquations
using ForwardDiff
using Optimization, OptimizationNLopt
using Random, DelimitedFiles

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
const modelfile	= false
const stimfile	= false
const sigdigs	= 5
const skipPars  = 0

# make the parameters for the solution
const Nt = 1000
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

# set up model parameters
P = zeros(Float64, 13 + 5*62)
lb= zeros(Float64, 13 + 5*62)
ub= zeros(Float64, 13 + 5*62)
#####		tsi,	tv1m,	tv2m,	tvp,	twm,	twp,	td,	to,	tr,	xk,	uc,	uv,	ucsi
P[1:13] .= [	29.0	19.6	1250.0	3.33	41.0	870.0	0.25	12.5	33.3	10.0	0.13	0.04	0.85	][1:13]
lb[1:13].= [ 	1.0, 	1.0, 	1.0, 	1.0, 	1.0, 	1.0, 	0.05, 	1.0, 	1.0, 	9.0, 	0.10, 	0.01, 	0.10 	][1:13]
ub[1:13].= [ 	1000.0, 1000.0, 2000.0, 15.0,	1000.0,	1000.0,	0.50, 	1000.0,	1000.0,	11.0,	0.15,	0.05,	0.90 	][1:13]

# initialize stimulus parameters with randomly chosen values, adapt lb/ub to fit
for m in 1:62
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

# model definition using (ensemble) ODE problem
function model1(θ,ensemble)
	prob = ODEProblem(fenkar!, u0, tspan, θ[1:16])
	
	function prob_func(prob, i, repeat)
		remake(prob, u0 = [target[1,1,i], θ[17+5*(i-1)], θ[18+5*(i-1)]], p = θ[[1:13; (14+5*(i-1)):(16+5*(i-1))]])
	end

	ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
	sim = solve(ensemble_prob, Tsit5(), ensemble, saveat = t, save_idxs=1:1, trajectories = 62, maxiters=Int(1e8))
	
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
		l = sum(abs2, (target[:,:,1:Nsols].-Array(sol))) * M(θ[1:13])
	end
	return l,sol
end

# initial loss and solution set
const l1,sol1 = loss(P,SciMLBase.NullParameters()); 

function plotFits(θ,sol; target=target)

	fig = Figure(resolution=(1800,1800))
	gax = fig[1,1] = GridLayout();
	axs = [Axis(gax[i,j]) for i in 1:8, j in 1:8];
	linkaxes!(axs...)
	for n in 1:Nsols
		# linear indexing into Array{Axis,2}
		lines!(axs[n], t, target[1,:,n], color=:black, linewidth=2)
		lines!(axs[n], t, Istim.(t,-θ[13+5*(n-1)+3],θ[13+5*(n-1)+1],θ[13+5*(n-1)+2]), linewidth=1.0, color=:red)
		lines!(axs[n], t, sol1[1,:,n], linewidth=1.5, color=:blue)
		lines!(axs[n], t, sol[1,:,n], linewidth=1.5, color=:orange)
	end
	for i in 1:7, j in 1:8
		hidexdecorations!(axs[i,j], ticks=false, grid = false)
	end
	for i in 1:8, j in 2:8
		hideydecorations!(axs[i,j], ticks=false, grid = false)
	end
	for i in (Nsols+1):(8*8)
		hidedecorations!(axs[i], ticks=true, grid = true)
	end
	colgap!(gax, 10)
	rowgap!(gax, 10)
	limits!([t[begin], t[end]], [-0.2, +1.2])
	fig
	save("./fittings/Nt_$(Nt)/all_fits_makie.pdf", fig);
	return nothing
end

plotFits(P,sol1);

