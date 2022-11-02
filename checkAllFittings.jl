include("./model.jl")
using .fenkar
using DifferentialEquations
using ForwardDiff
using Optimization, OptimizationNLopt
using Random, DelimitedFiles
using Dierckx
using Statistics

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
const Nt = parse(Int, ARGS[1])
const dt = 2.0			# time between samples -- set by the data
const tspan = (0.0,Nt*dt)
const t = collect(range(tspan[1], tspan[2]; length=Nt));
const u0 = rand(Float64,3)

const Nsols = 62
const target = zeros(Float64, 1, Nt, Nsols);
const t0s = zeros(Float64, Nsols);
const BCLs = zeros(Float64, Nsols);

target .= reshape(readdlm("./data/Nt_$(Nt)/target.txt"), size(target));
t0s .= reshape(readdlm("./data/Nt_$(Nt)/t0s.txt"), size(t0s));
BCLs .= reshape(readdlm("./data/Nt_$(Nt)/BCLs.txt"), size(BCLs));

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

function analyzeTraces(t, Vs)
	APDs = [];
	DIs  = [];
	APAs = [];
	for n in 1:size(Vs,3)
		APD, DI, APA = analyzeTrace(t, Vs[1,:,n][:]);
		push!(APDs, APD);
		push!(DIs,   DI);
		push!(APAs, APA);
	end
	return (APDs, DIs, APAs)
end

function analyzeTrace(t, V; V90=0.2)
	dt = mean(diff(t))
	Vt = Spline1D(t[:], V.-V90, k=3);
	
	R = roots(Vt; maxn=Int(5e3));	# time points R: V(R) == V90
	
	# storage for APD, DI, APA for this BCL
	APD = Float64[]
	DI  = Float64[]
	APA = Float64[]
	
	if !isempty(R)
		D = derivative(Vt, R);			# V'(R)
		
		for n in 1:length(R)-1
			if D[n] > 0 && D[n+1] < 0
				push!(APD, R[n+1]-R[n])
				push!(APA, V90+maximum(Vt(R[n]:dt:R[n+1])))
			elseif D[n] < 0 && D[n+1] > 0
				push!(DI, R[n+1]-R[n])
			end
		end
	end
	return (APD, DI, APA)
end

function distros(data, LL)
	pnames=["tsi", "tv1m", "tv2m", "tvp", "twm", "twp", "td", "to", "tr", "xk", "uc", "uv", "ucsi"]
	
	fig, axs = plt.subplots(1,length(pnames),figsize=(dw+sw,sw/2),sharey=true,constrained_layout=true)
	for m in 1:length(pnames)
		if m==length(pnames)
			sc = axs[m].scatter(data[m,:], LL, c=LL, s=10, cmap="viridis")
			fig.colorbar(sc, ax=axs[:], aspect=50, label="Loss")
		else
			axs[m].scatter(data[m,:], LL, c=LL, s=10, cmap="viridis");
		end
		axs[m].set_xlabel("$(pnames[m])")
		if m==1
			axs[m].set_ylabel("Loss")
		end
	end
	fig.savefig("./fittings/Nt_$(Nt)/distros_pp.pdf",bbox_inches="tight",dpi=300)
	plt.close(fig)
end

function paramCovariance(data, LL)
	
	pnames=["tsi", "tv1m", "tv2m", "tvp", "twm", "twp", "td", "to", "tr", "xk", "uc", "uv", "ucsi"]
	fig, axs = plt.subplots(length(pnames),length(pnames),figsize=(dw*1.2,dw),sharey="row",sharex="col",constrained_layout=true)
	
	weightedMeans = [sum(data[n,:].*(1.0./LL[:])) for n in 1:length(pnames)]/sum(1.0./LL[:])
	
	for m in 1:length(pnames), n in 1:length(pnames)
		if m==length(pnames) && n==length(pnames)
			sc = axs[n,m].scatter(data[m,:], data[n,:], c=LL, s=10, cmap="viridis")
			fig.colorbar(sc, ax=axs[:], aspect=50, label="Loss")
		else
			axs[n,m].scatter(data[m,:], data[n,:], c=LL, s=10, cmap="viridis");
		end
		if m==1
			axs[n,m].set_ylabel("$(pnames[n])")
		end
		if n==length(pnames)
			axs[n,m].set_xlabel("$(pnames[m])")
		end
		axs[n,m].plot(weightedMeans[m], weightedMeans[n], ".r", markersize=4)
		axs[n,m].tick_params(direction="in")
	end
	fig.savefig("./fittings/Nt_$(Nt)/covariance_pp.pdf",bbox_inches="tight",dpi=300)
	plt.close(fig)
end

function plotBCLs(BCLs, APDs, DIs, APAs)

	fig, axs = plt.subplots(3,1, figsize=(sw,dw), sharex=true, constrained_layout=true);
	#axs[1].set_xlabel("BCL [ms]")
	axs[1].set_ylabel("APD [ms]")
	#axs[2].set_xlabel("BCL [ms]")
	axs[2].set_ylabel("DI  [ms]")
	axs[3].set_xlabel("BCL [ms]")
	axs[3].set_ylabel("APA [  ]")
	axs[1].plot(APDs[:,1], APDs[:,2], ".C0", markersize=4, alpha=0.3, label="")
	axs[2].plot( DIs[:,1],  DIs[:,2], ".C0", markersize=4, alpha=0.3, label="")
	axs[3].plot(APAs[:,1], APAs[:,2], ".C0", markersize=4, alpha=0.3, label="")
	APDs, DIs, APAs = analyzeTraces(t, target);
	for (BCL, APD, DI, APA) in zip(BCLs, APDs, DIs, APAs)
		axs[1].plot(BCL*ones(Float64, length(APD)), APD, ".k", markersize=6)
		axs[2].plot(BCL*ones(Float64, length(DI)),   DI, ".k", markersize=6)
		axs[3].plot(BCL*ones(Float64, length(APA)), APA, ".k", markersize=6)
	end
	plt.savefig("./fittings/Nt_$(Nt)/BCLs.pdf",bbox_inches="tight")
	plt.close(fig)
	return nothing
end

function main(;truncateModelParams=false)
	
	# get known parameters and form deflation operator
	PP = knownParameters();
	LL = knownLosses();

	APDs = []
	DIs  = []
	APAs = []
	for (n,(P,L)) in enumerate(zip(PP,LL))
		# get loss and solution for parameters P
		l,sol = loss(P,SciMLBase.NullParameters()); 
		tmpAPDs, tmpDIs, tmpAPAs = analyzeTraces(t, sol);
		
		for n in 1:Nsols
			for m in eachindex(tmpAPDs[n])
				push!(APDs, [BCLs[n] ;; tmpAPDs[n][m]]);
			end
			for m in eachindex(tmpDIs[n])
				push!(DIs,  [BCLs[n] ;; tmpDIs[n][m]]);
			end
			for m in eachindex(tmpAPAs[n])
				push!(APAs, [BCLs[n] ;; tmpAPAs[n][m]]);
			end
		end
				
		# plotFits(P,sol,n; target=target);
		
		if l > 1.05*L
			print("Oddity: parameter set $n; \tL=$(L), \tl=$(l).\n")
		end
	end
	
	APDs = reduce(vcat, APDs);
	 DIs = reduce(vcat,  DIs);
	APAs = reduce(vcat, APAs);
	
	plotBCLs(BCLs, APDs, DIs, APAs);
	
	data = reduce(hcat, PP);
	data = Float64.(data);
	
	# optionally truncate to just the model parameters
	if truncateModelParams
		data = data[1:13,:];
	end

	distros(data, LL);
	paramCovariance(data, LL);
	
	n = argmin(LL);
	print("\nLowest per-element-loss = $(sqrt(LL[n]/Nt/Nsols)) for index $(n).\n")
	
	return nothing
end
main()