using DelimitedFiles, StatsBase, TSne, UMAP, Plots

const Nt = parse(Int, ARGS[1]);

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

function main(;truncateModelParams=true)
	LL = knownLosses();
	PP = knownParameters();
	
	data = reduce(hcat, PP);
	data = Float64.(data);
	
	# optionally truncate to just the model parameters
	if truncateModelParams
		data = data[1:13,:];
	end
	
	distros(data, LL);
	paramCovariance(data, LL);
	
	dim = 2;
	neighbors = min(50, length(LL)-1);
	epochs = 1000;
	
	Y = tsne(collect(transpose(data)), dim, 0, epochs, neighbors)
	theplot = scatter(Y[:,1], Y[:,2], zcolor=LL, marker=(:viridis, 5), label="Loss")
	savefig(theplot, "./fittings/Nt_$(Nt)/tSNE.pdf")
	
	## UMAP.jl seems to have a linker problem on Apple Silicon
	embedding = umap(data, dim; n_neighbors=neighbors, n_epochs=epochs )
	theplot = scatter(embedding[1,:], embedding[2,:], zcolor=LL, marker=(:viridis, 5), label="Loss")
	savefig(theplot, "./fittings/Nt_$(Nt)/UMAP.pdf")
		
	return nothing
end

main()
