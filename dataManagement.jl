__precompile__()
module dataManagement

using DelimitedFiles

export writeParams, appendLosses, saveProgress, knownParameters, loadTargetData!, knownFits, reviseFits

const sigdigs = 10

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

function loadTargetData!(target, t0s, BCLs; Nt=1)
	target .= reshape(readdlm("./data/Nt_$(Nt)/target.txt"), size(target));
	t0s .= reshape(readdlm("./data/Nt_$(Nt)/t0s.txt"), size(t0s));
	BCLs .= reshape(readdlm("./data/Nt_$(Nt)/BCLs.txt"), size(BCLs));
	return nothing
end

# write fit parameters to file(s)
function writeParams(θ, l, filename; writeModelFile = false, writeStimFile = false )
	open(filename, "w") do io
		write(io, "# Loss = $(l)\n\n")
		write(io, "# tsi\t\ttv1m\t\ttv2m\t\ttvp\t\ttwm\t\ttwp\t\ttd\t\tto\t\ttr\t\txk\t\tuc\t\tuv\t\tucsi\n")
		writedlm(io, transpose(round.(θ[1:13],sigdigits=sigdigs)))
		write(io,"\n")
		write(io, "# t0\t\tTI\t\tIA\t\tv0\t\tw0\n")
		writedlm(io, transpose(reshape(round.(θ[14:end],sigdigits=sigdigs-1), 5, :)))
	end
	if writeModelFile
		open("./fittings/Nt_$(Nt)/model_params.txt", "w") do io
			writedlm(io, transpose(round.(θ[1:13],sigdigits=sigdigs)))
		end
	end
	if writeStimFile
		open("./fittings/Nt_$(Nt)/stim_params.txt", "w") do io
			writedlm(io, transpose(reshape(round.(θ[14:end],sigdigits=sigdigs-1), 5, :)))
		end
	end
	return nothing
end

# append losses to filename
function appendLosses(P1, P2, filename)
	open(filename, "a") do io
		write(io, "\n")
		write(io, "# Loss (initial) = $(loss(P1,nothing)[1])\n")
		write(io, "# Loss (final) = $(loss(P2,nothing)[1])\n")
		Q = P2; Q[1:13] .= P[1:13];	
		write(io, "# Loss (resample) = $(loss(Q,nothing)[1])\n")
	end
	return nothing
end

# save progress during optimization
function saveProgress(θ, l, sol, Nt; target=target, plotting=false)
	writeParams(θ,l,"./fittings/Nt_$(Nt)/all_params.txt");
	if plotting
		fig, axs = plotFits(θ, sol; target=target)
		fig.savefig("./fittings/Nt_$(Nt)/all_fits.pdf",bbox_inches="tight")
		plt.close(fig)
	end
	return nothing
end

function knownParameters(Nt; truncate=true)
	PP = []
	LL = []
	loading = true
	ind = 0
	while loading
		try
			ind = ind + 1
			tmp = readdlm("./fittings/Nt_$(Nt)/$(ind).txt"; comments=true, comment_char='#');
			if truncate
				push!(PP, tmp[1,1:13][:])
			else
				push!(PP, vcat(tmp[1,1:13][:],transpose(tmp[2:end,1:5])[:]))
			end
		catch
			loading = false
		end
	end
	print("There are $(length(PP)) known model parameter sets!\n");
	return PP
end

function knownLosses(Nt)
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


function knownFits(Nt; truncate=true)
	LL = []; PP = []
	ind = 0; loading = true; 
	while loading
		try
			ind = ind + 1
			tmp = readdlm("./fittings/Nt_$(Nt)/$(ind).txt");
			loss = Float64(tmp[1,4])
			push!(LL, loss)
			tmp = readdlm("./fittings/Nt_$(Nt)/$(ind).txt"; comments=true, comment_char='#');
			if truncate
				push!(PP, tmp[1,1:13][:])
			else
				push!(PP, vcat(tmp[1,1:13][:],transpose(tmp[2:end,1:5])[:]))
			end
		catch
			loading = false
		end
	end
	print("There are $(length(LL)) known model losses!\n");
	return LL,PP
end

function reviseFits(Nt, ll; truncate=false)
	
	LL, PP = knownFits(Nt; truncate=truncate);
	ls = []
	sols = []
	for (n,(P,L)) in enumerate(zip(PP,LL))
		# get loss and solution for parameters P
		l,sol = ll(P);
		push!(ls, l);
		push!(sols,sol);
		if abs(l-L) > 0.05*min(L,l)
			filename = "./fittings/Nt_$(Nt)/$(n).txt"
			print("Oddity: parameter set $n; \tL=$(L), \tl=$(l).\n")
			print("Rewriting $(filename)...\t");
			writeParams(P, l, filename);
			LL[n] = l;
			print(" Done.\n")
		end
	end
	return ls,sols
end

end