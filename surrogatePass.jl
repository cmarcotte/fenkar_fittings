include("./model.jl")
using .fenkar
using DifferentialEquations
using ForwardDiff
using Optimization, OptimizationNLopt
using Random, DelimitedFiles
using Dierckx
using Statistics
using Surrogates
using BenchmarkTools

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

#using Plots
	
# make the parameters for the solution
const Nt = 500
const dt = 2.0			# time between samples -- set by the data
const tspan = (0.0,Nt*dt)
const t = collect(range(tspan[1], tspan[2]; length=Nt));
const u0 = rand(Float64,3)

const Nsols = 62
const target = zeros(Float64, 1, Nt, Nsols);
const t0s = zeros(Float64, Nsols);
const BCLs = zeros(Float64, Nsols);

function knownLosses()
	LL = []
	loading = true
	ind = 0
	while loading
		try
			ind = ind + 1
			tmp = readdlm("./fittings/Nt_$(Nt)/$(ind).txt");
			loss = Float64(tmp[1,4])
			push!(LL, loss)
		catch
			loading = false
		end
	end
	print("There are $(length(LL)) known model losses!\n");
	return LL
end

function knownParameters()
	PP = []
	loading = true
	ind = 0
	while loading
		try
			ind = ind + 1
			tmp = readdlm("./fittings/Nt_$(Nt)/$(ind).txt"; comments=true, comment_char='#');
			push!(PP, vcat(tmp[1,1:13][:],transpose(tmp[2:end,1:5])[:]))
		catch
			loading = false
		end
	end
	print("There are $(length(PP)) known model parameter sets!\n");
	return PP
end

# model definition using (ensemble) ODE problem
function model1(θ,ensemble)
	prob = ODEProblem(fenkar!, u0, tspan, θ[1:16])
	
	function prob_func(prob, i, repeat)
		remake(prob, u0 = [target[1,1,i], θ[17+5*(i-1)], θ[18+5*(i-1)]], p = θ[[1:13; (14+5*(i-1)):(16+5*(i-1))]])
	end

	ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
	sim = solve(ensemble_prob, Tsit5(), ensemble, saveat = t, save_idxs=1:1, trajectories = 62, maxiters=Int(1e8), abstol=1e-8, reltol=1e-6)
	
end

# define loss function
function loss(θ, _p; ensemble=EnsembleThreads())
	sol = model1(θ,ensemble)
	if any((s.retcode != :Success for s in sol))
		l = Inf
	elseif size(Array(sol)) != size(target[:,:,1:Nsols])
		print("I'm a doodoohead poopface") 
	else
		l = sum(abs2, (target[:,:,1:Nsols].-Array(sol)))
	end
	return l,sol
end

function plotFits(θ,sol,index; target=target)

	fig, axs = plt.subplots(Int(ceil(Nsols/8)),8, figsize=(dw,dw*1.05*(Int(ceil(Nsols/8))/8)), 
				sharex=true, sharey=true, constrained_layout=true)
	for n in 1:Nsols
		# linear indexing into Array{Axis,2}
		axs[n].plot(t, target[1,:,n], "-k", linewidth=1.6)
		# plotting the (negated) stimulus current to make sure things line up
		axs[n].plot(t, Istim.(t,-θ[13+5*(n-1)+3],θ[13+5*(n-1)+1],θ[13+5*(n-1)+2]), "-r", linewidth=0.5)
		axs[n].plot(t, sol[1,:,n], "-C1", linewidth=1)
	end
	xt = collect(0:4).*(Nt*dt/4);
	axs[1].set_ylim([-0.1,1.1])
	axs[1].set_xlim([t[begin],t[end]])
	axs[1].set_xticks(xt)
	axs[1].set_xticklabels(["","$(round(xt[2]))","","$(round(xt[4]))",""])
	plt.savefig("./fittings/Nt_$(Nt)/$index.pdf",bbox_inches="tight")
	plt.close(fig)
		
	return nothing
end

function main(;truncateModelParams=false)
	
	# get known parameters and form deflation operator
	PP = knownParameters();
	LL = knownLosses();
		
	data = reduce(hcat, PP);
	data = Float64.(data);
	
	# optionally truncate to just the model parameters
	if truncateModelParams
		data = data[1:13,:];
	end

	target .= reshape(readdlm("./data/Nt_$(Nt)/target.txt"), size(target));
	t0s .= reshape(readdlm("./data/Nt_$(Nt)/t0s.txt"), size(t0s));
	BCLs .= reshape(readdlm("./data/Nt_$(Nt)/BCLs.txt"), size(BCLs));
	
	# set up model parameters
	lb= zeros(Float64, 13 + 5*62)
	ub= zeros(Float64, 13 + 5*62)
	#####			tsi,	tv1m,	tv2m,	tvp,	twm,	twp,	td,		to,		tr,		xk,		uc,		uv,		ucsi
	lb[1:13].= [ 	1.0, 	1.0, 	1.0, 	1.0, 	1.0, 	1.0, 	0.05, 	1.0, 	1.0, 	9.0, 	0.10, 	0.01, 	0.10 	][1:13]
	ub[1:13].= [ 	1000.0, 1000.0, 2000.0, 15.0,	1000.0,	1000.0,	0.50, 	1000.0,	1000.0,	11.0,	0.15,	0.05,	0.90 	][1:13]
	
	# initialize stimulus parameters with randomly chosen values, adapt lb/ub to fit
	for m in 1:Nsols	
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
	
	for n in 1:length(ub)
		lb[n] = max(lb[n], minimum(data[n,:]))
		ub[n] = min(ub[n], maximum(data[n,:]))
	end
	
#=
	LL = Float64.(LL)
	QQ = [Tuple(Float64.(P)) for P in PP]
	kriging_surrogate = Kriging(QQ, LL, lb, ub)
	
	# new test params
	P = lb .+ rand(Float64,length(ub)).*(ub.-lb);
	kriging_loss = kriging_surrogate(P)
	
	loss(Q) = loss(Q, nothing)[1]
	true_loss = loss(P)
	@show surrogate_optimize(loss, SRBF(), lb, ub, kriging_surrogate, SobolSample())
=#	
	
	loss(Q) = loss(Q, nothing)[1]
	QQ = vcat(sample(1000,lb,ub,SobolSample()), [Tuple(Float64.(P)) for P in PP])
	LL = loss.(QQ)
	
	kriging_surrogate = Kriging(QQ, LL, lb, ub)

    @benchmark kriging_surrogate(P)

    @benchmark loss(P)

	#P = lb .+ rand(Float64,length(ub)).*(ub.-lb);
	P = QQ[argmin(LL)] .+ 0.1.*abs.(QQ[argmin(LL)]).*randn(Float64,length(ub))
	true_loss = loss(P)
	kriging_loss = kriging_surrogate(P)
	
	@show surrogate_optimize(loss, SRBF(), lb, ub, kriging_surrogate, SobolSample())
	
	return nothing
end
main()