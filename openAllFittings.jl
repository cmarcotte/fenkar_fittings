using DelimitedFiles, Plots, StatsBase, TSne, UMAP

const Nt = 1200;

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
		push!(plts, scatter(data[m,:], LL,
					marker = (:circle, 5, 0.3, stroke(0, 0, :black, :dot)), 
					xlabel="$(pnames[m])", ylabel=(m%5==1 ? "Loss" : ""), legend=false, dpi=300));
	end
	allplts = plot(plts...; layout=(3,5), size=(1600,1000), Nt_$(Nt)=:y, dpi=300)
	savefig(allplts, "./fittings/Nt_$(Nt)/distros.pdf")
end

function main()
	LL = knownLosses();
	PP = knownParameters();
	
	data = reduce(hcat, PP);
	#data = data[1:13,:];
	data = Float64.(data);
	
	distros(data, LL);
		
	Y = tsne(collect(transpose(data)), 2, 0, 20000, 50)
	theplot = scatter(Y[:,1], Y[:,2], zcolor=LL, marker=(:viridis, 5 .+ 5 .*sqrt.(LL./(500*62))), label="Loss")
	savefig(theplot, "./fittings/Nt_$(Nt)/tSNE.pdf")
	
	## UMAP.jl seems to have a Nt_$(Nt)er problem on Apple Silicon
	embedding = umap(data, 2; n_neighbors=50 )
	theplot = scatter(embedding[1,:], embedding[2,:], zcolor=LL, marker=(:viridis, 5 .+ 5 .*sqrt.(LL./(500*62))), label="Loss")
	savefig(theplot, "./fittings/Nt_$(Nt)/UMAP.pdf")
	
	return nothing
end

main()