__precompile__()
module plotting

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

export plotFits, plotAllFits, distros, paramCovariance

function plotFits(t, θ, sol; Nt=Nt, Nsols=Nsols, target=target)

	fig, axs = plt.subplots(Int(ceil(Nsols/8)),8, figsize=(dw,dw*1.05*(Int(ceil(Nsols/8))/8)), 
				sharex=true, sharey=true, constrained_layout=true)
	xt = collect(0:4).*(Nt*dt/4);
	for n in 1:Nsols
		# linear indexing into Array{Axis,2}
		axs[n].plot(t, target[1,:,n], "-k", linewidth=1.6)
		# plotting the (negated) stimulus current to make sure things line up
		axs[n].plot(t, Istim.(t,-θ[13+5*(n-1)+3],θ[13+5*(n-1)+1],θ[13+5*(n-1)+2]), "-r", linewidth=0.5)
		axs[n].plot(t, sol1[1,:,n], "-C0", linewidth=1)
		axs[n].plot(t, sol[1,:,n], "-C1", linewidth=1)
	end
	axs[1].set_ylim([-0.1,1.1])
	axs[1].set_xlim([t[begin],t[end]])
	axs[1].set_xticks(xt)
	axs[1].set_xticklabels(["","$(round(xt[2]))","","$(round(xt[4]))",""])
	plt.savefig("./fittings/Nt_$(Nt)/$index.pdf",bbox_inches="tight")
	plt.close(fig)
		
	return nothing
end

function plotAllFits(t,ls,sols; Nt=Nt, Nsols=Nsols, target=target)

	fig, axs = plt.subplots(Int(ceil(Nsols/8)),8, figsize=(dw,dw*1.05*(Int(ceil(Nsols/8))/8)), 
				sharex=true, sharey=true, constrained_layout=true)
	for n in 1:Nsols
		# linear indexing into Array{Axis,2}
		axs[n].plot(t, target[1,:,n], "-k", linewidth=1.6)
	end
	for (l,sol) in zip(ls,sols)
		for n in 1:Nsols
			axs[n].plot(t, sol[1,:,n], "-C1", linewidth=0.2, alpha=sqrt(l)/maximum(sqrt.(ls)))
		end
	end
	xt = collect(0:4).*(t[end]/4);
	axs[1].set_ylim([-0.1,1.1])
	axs[1].set_xlim([t[begin],t[end]])
	axs[1].set_xticks(xt)
	axs[1].set_xticklabels(["","$(round(xt[2]))","","$(round(xt[4]))",""])
	plt.savefig("./fittings/Nt_$(Nt)/all_fits.pdf",bbox_inches="tight")
	plt.close(fig)
		
	return nothing
end

function distros(data, LL, Nt)
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

function losses(LL)
	fig = plt.figure(figsize=(sw,sw/2),constrained_layout=true)
	plt.hist(LL, bins=33, density=true, label="Loss")
	#plt.xlabel("\$ \ell \$")
	#plt.ylabel("\$ P(\ell) \$")
	fig.savefig("./fittings/Nt_$(Nt)/losses.pdf",bbox_inches="tight",dpi=300)
	plt.close(fig)
end

function paramCovariance(data, LL, Nt)
	
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

function plotBCLs(BCLs, APDs, DIs, APAs; Nt=nt)

	fig, axs = plt.subplots(3,1, figsize=(sw,dw), sharex=true, constrained_layout=true);
	#axs[1].set_xlabel("BCL [ms]")
	axs[1].set_ylabel("APD [ms]")
	#axs[2].set_xlabel("BCL [ms]")
	axs[2].set_ylabel("DI  [ms]")
	axs[3].set_xlabel("BCL [ms]")
	axs[3].set_ylabel("APA [  ]")
	sc = axs[1].scatter(APDs[:,1], APDs[:,3], c=APDs[:,2], s=4, alpha=0.3, cmap="viridis", label="")
	axs[2].scatter( DIs[:,1],  DIs[:,3], c= DIs[:,2], s=4, alpha=0.3, cmap="viridis", label="")
	axs[3].scatter(APAs[:,1], APAs[:,3], c=APAs[:,2], s=4, alpha=0.3, cmap="viridis", label="")
	fig.colorbar(sc, ax=axs[:], aspect=50, label="Loss")
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

end