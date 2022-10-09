using DelimitedFiles, Plots, StatsBase, TSne, UMAP

function knownLosses()
	LL = []
	loading = true
	ind = 0
	while loading
		try
			ind = ind + 1
			tmp = readdlm("./fittings/$(ind).txt");
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
			tmp = readdlm("./fittings/$(ind).txt"; comments=true, comment_char='#');
			push!(PP, vcat(tmp[1,1:13][:],transpose(tmp[2:end,1:5])[:]))
		catch
			loading = false
		end
	end
	print("There are $(length(PP)) known model parameter sets!\n");
	return PP
end

function histos(data)
	plts = []
	pnames=["tsi", "tv1m", "tv2m", "tvp", "twm", "twp", "td", "to", "tr", "xk", "uc", "uv", "ucsi"]
	for m in 1:13
		push!(plts, scatter(data[m+1,:], data[1,:], 
					marker = (:circle, 5, 0.3, stroke(0, 0, :black, :dot)), 
					xlabel="$(pnames[m])", ylabel=(m%5==0 ? "Loss" : ""), legend=false, dpi=300));
	end
	allplts = plot(plts...; layout=(3,5), size=(1600,1000), link=:y, dpi=300)
	savefig(allplts, "./fittings/histos.pdf")
end

function main()
	LL = knownLosses();
	PP = knownParameters();
	
	data = reduce(hcat, PP);
	data = vcat(transpose(LL),data); 
	data = data[1:14,:];
	data = Float64.(data);
	
	histos(data);
		
	Y = tsne(collect(transpose(data)), 2, 0, 20000, 50)
	theplot = scatter(Y[:,1], Y[:,2], zcolor=LL, marker=(:viridis, 5 .+ 5 .*sqrt.(LL./(500*62))), label="Loss")
	savefig(theplot, "./fittings/tSNE.pdf")
	
	## UMAP.jl seems to have a linker problem on Apple Silicon
	#=
	embedding = umap(data, 2; n_neighbors=50 )
	theplot = scatter(embedding[1,:], embedding[2,:], zcolor=LL, marker=(:viridis, 5 .+ 5 .*sqrt.(LL./(500*62))), label="Loss")
	savefig(theplot, "./fittings/UMAP.pdf")
	=#
	return nothing
end

main()