using DelimitedFiles, Plots, StatsBase, TSne

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

function main()
	LL = knownLosses();
	ll = (LL.-minimum(LL))./(maximum(LL).-minimum(LL));
	PP = knownParameters();
	
	data = reduce(hcat, PP);
	data = collect(transpose(Float64.(data[1:13,:])));
	
	Y = tsne(data, 2, 0, 100000, 30)
	
	theplot = scatter(Y[:,1], Y[:,2], zcolor=LL, marker=(:viridis, 5 .+ 5 .*sqrt.(LL./(500*62))), label="Loss")
	savefig(theplot, "./fittings/tSNE.pdf")
end

main()