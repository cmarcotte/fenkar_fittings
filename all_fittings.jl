using DelimitedFiles, StatsBase, TSne, UMAP, Printf, Plots, Plots.PlotMeasures, StatsPlots

function knownFits()
	LL = []; PP = []
	ind = 0; loading = true; 
	while loading
		try
			ind = ind + 1
			tmp = readdlm(@sprintf("./fittings/all/%05d.txt", ind));
			loss = Float64(tmp[1,4])
			push!(LL, loss)
			tmp = readdlm(@sprintf("./fittings/all/%05d.txt", ind); comments=true, comment_char='#');
			push!(PP, vcat(tmp[1,1:13][:],transpose(tmp[2:end,1:5])[:]))
		catch
			loading = false
		end
	end
	print("There are $(length(LL)) known model losses!\n");
	return LL,PP
end

function main()
	LL,PP = knownFits();
	
	data = reduce(hcat, PP);
	data = Float64.(data);
	
	pnames = ["tsi", "tv1m", "tv2m", "tvp", "twm", "twp", "td", "to", "tr", "xk", "uc", "uv", "ucsi"]
	pp = [];
	for n in 1:13
		push!(pp, density(data[n,:], yticks=[], legend=false, normalize=true, xlabel=pnames[n]))
	end
	pa = plot(pp..., layout=(1,13), size=(1700,600))
	p2 = plot(); for n in (13+1):5:323; density!(p2, data[n,:], yticks=[], legend=false, normalize=true); end; plot!(p2, xlabel="t_0");
	p3 = plot(); for n in (13+2):5:323; density!(p3, data[n,:], yticks=[], legend=false, normalize=true); end; plot!(p3, xlabel="BCL");
	p4 = plot(); density!(p4, data[(13+3):5:323,:][:], yticks=[], legend=false, normalize=true, xlabel="I_{stim}");
	p5 = plot(); density!(p5, data[(13+4):5:323,:][:], yticks=[], legend=false, normalize=true, xlabel="v_0"); 
	p6 = plot(); density!(p6, data[(13+5):5:323,:][:], yticks=[], legend=false, normalize=true, xlabel="w_0");
	pb = plot(p2,p3,p4,p5,p6, layout=(1,5), size=(1700,300))
	plot(pa,pb, layout=(2,1), size=(1700,1000), bottom_margin=10mm)
	savefig("fittings/distros.pdf")
	
	dim = 2;
	neighbors = min(50, length(LL)-1);
	epochs = 40_000;
	
	# UMAP.jl seems to have a linker problem on native Julia on Apple Silicon
	embedding = umap(data, dim; n_neighbors=neighbors, n_epochs=epochs )
	theplot = scatter(embedding[1,:], embedding[2,:], zcolor=LL, marker=(:viridis, 5), label="Loss", xticks=[], yticks=[])
	savefig(theplot, "./fittings/UMAP.pdf")
	
	Y = tsne(collect(transpose(data)), dim, 0, epochs, neighbors)
	theplot = scatter(Y[:,1], Y[:,2], zcolor=LL, marker=(:viridis, 5), label="Loss", xticks=[], yticks=[])
	savefig(theplot, "./fittings/tSNE.pdf")
	
	return nothing
end

main()