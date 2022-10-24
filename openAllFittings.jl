using DelimitedFiles, StatsBase, TSne, UMAP
using Plots
#=
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
=#
const Nt = 1000;

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

function distros(data, LL)
	plts = []
	pnames=["tsi", "tv1m", "tv2m", "tvp", "twm", "twp", "td", "to", "tr", "xk", "uc", "uv", "ucsi"]
	for m in 1:13
		push!(plts, scatter(data[m,:], LL, zcolor=LL,
					marker = (:circle, 5, Plots.stroke(0, :black)), 
					xlabel="$(pnames[m])", ylabel=(m%5==1 ? "Loss" : ""), legend=false, dpi=300));
	end
	allplts = plot(plts...; layout=(3,5), size=(1600,1000), link=:y, dpi=300)
	savefig(allplts, "./fittings/Nt_$(Nt)/distros.pdf")
end
#=
function distros(data, LL)
	pnames=["tsi", "tv1m", "tv2m", "tvp", "twm", "twp", "td", "to", "tr", "xk", "uc", "uv", "ucsi"]
	
	fig, axs = plt.subplots(1,length(pnames),figsize=(dw+sw,sw/2),sharey=true,constrained_layout=true)
	for m in 1:length(pnames)
		if m==length(pnames)
			sc = axs[m].scatter(data[m,:], LL, c=LL, cmap="viridis")
			fig.colorbar(sc, ax=axs[:], aspect=50, label="Loss")
		else
			axs[m].scatter(data[m,:], LL, c=LL);
		end
		axs[m].set_xlabel("$(pnames[m])")
		if m==1
			axs[m].set_ylabel("Loss")
		end
	end
	fig.savefig("./fittings/Nt_$(Nt)/distros.pdf",bbox_inches="tight",dpi=300)
	plt.close(fig)
end
=#

function paramCovariance(data, LL)
	plts = []
	pnames=["tsi", "tv1m", "tv2m", "tvp", "twm", "twp", "td", "to", "tr", "xk", "uc", "uv", "ucsi"]
	for n in 1:9,m in 1:9
		push!(plts, scatter(data[m,:], data[n,:],
					zcolor=LL, marker = (:viridis, :circle, 5, Plots.stroke(0, :black)), 
					xlabel=(n==9 ? "$(pnames[m])" : ""), 
					ylabel=(m==1 ? "$(pnames[n])" : ""),
					framestyle=((n<9 || m>1) ? :grid : :semi),
					legend=false, dpi=300));
	end
	allplts = plot(plts...; layout=(9,9), size=(1200,1200), linky=:row, linkx=:col, dpi=300)
	savefig(allplts, "./fittings/Nt_$(Nt)/covariance.pdf")
end
#=
function paramCovariance(data, LL)
	
	pnames=["tsi", "tv1m", "tv2m", "tvp", "twm", "twp", "td", "to", "tr"]#, "xk", "uc", "uv", "ucsi"]
	fig, axs = plt.subplots(length(pnames),length(pnames),figsize=(dw*1.2,dw),sharey="row",sharex="col",constrained_layout=true)
	for n in 1:length(pnames), m in 1:length(pnames)
		if m==length(pnames) && n==length(pnames)
			sc = axs[n,m].scatter(data[m,:], data[n,:], c=LL, cmap="viridis")
			fig.colorbar(sc, ax=axs[:], aspect=50, label="Loss")
		else
			axs[n,m].scatter(data[m,:], data[n,:], c=LL);
		end
		if m==1
			axs[n,m].set_ylabel("$(pnames[n])")
		end
		if n==length(pnames)
			axs[n,m].set_xlabel("$(pnames[m])")
		end
	end
	fig.savefig("./fittings/Nt_$(Nt)/covariance.pdf",bbox_inches="tight",dpi=300)
	plt.close(fig)
end
=#
function main(;truncateModelParams=true)
	LL = knownLosses();
	PP = knownParameters();
	
	data = reduce(hcat, PP);
	data = Float64.(data);
	
	# optionally truncate to just the model parameters
	if truncateModelParams
		data = data[1:13,:];
	end
	
	#distros(data, LL);
	#paramCovariance(data, LL);
		
	Y = tsne(collect(transpose(data)), 2, 0, 500000, min(50, length(LL)-1))
	theplot = scatter(Y[:,1], Y[:,2], zcolor=LL, marker=(:viridis, 5), label="Loss")
	savefig(theplot, "./fittings/Nt_$(Nt)/tSNE.pdf")
	
	## UMAP.jl seems to have a linker problem on Apple Silicon
	embedding = umap(data, 2; n_neighbors=min(50, length(LL)-1) )
	theplot = scatter(embedding[1,:], embedding[2,:], zcolor=LL, marker=(:viridis, 5), label="Loss")
	savefig(theplot, "./fittings/Nt_$(Nt)/UMAP.pdf")
	
	n = argmin(LL);
	print("\nLowest per-element-loss = $(LL[n]/Nt) for index $(n).\n")
	
	return nothing
end

main()