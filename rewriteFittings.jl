include("./model.jl")
using .fenkar
using OrdinaryDiffEq
using ForwardDiff, Optimization, OptimizationNLopt
using Random, DelimitedFiles, Statistics

const sigdigs = 10

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
		l = sum(abs2, (target[:,:,1:Nsols].-Array(sol))) / (Nt*Nsols)
	end
	return l,sol
end

function writeParams(θ, l, filename)
	open(filename, "w") do io
		write(io, "# Loss = $(l)\n\n")
		write(io, "# tsi\t\ttv1m\t\ttv2m\t\ttvp\t\ttwm\t\ttwp\t\ttd\t\tto\t\ttr\t\txk\t\tuc\t\tuv\t\tucsi\n")
		writedlm(io, transpose(round.(θ[1:13],sigdigits=sigdigs)))
		write(io,"\n")
		write(io, "# t0\t\tTI\t\tIA\t\tv0\t\tw0\n")
		writedlm(io, transpose(reshape(round.(θ[14:end],sigdigits=sigdigs-1), 5, :)))
	end
	open("./fittings/Nt_$(Nt)/model_params.txt", "w") do io
		writedlm(io, transpose(round.(θ[1:13],sigdigits=sigdigs)))
	end
	open("./fittings/Nt_$(Nt)/stim_params.txt", "w") do io
		writedlm(io, transpose(reshape(round.(θ[14:end],sigdigits=sigdigs-1), 5, :)))
	end
	return nothing;
end


function main(;truncateModelParams=false)
	
	# get known parameters and form deflation operator
	PP = knownParameters();
	LL = knownLosses();

	for (n,(P,L)) in enumerate(zip(PP,LL))
		# get loss and solution for parameters P
		l,sol = loss(P,SciMLBase.NullParameters()); 
		
		if abs(l-L) > 0.05*min(L,l)
			filename = "./fittings/Nt_$(Nt)/$(n).txt"
			print("Oddity: parameter set $n; \tL=$(L), \tl=$(l).\n")
			print("Rewriting $(filename)...\t");
			writeParams(P, l, filename);
			LL[n] = l;
			print(" Done.\n")
		end
	end
		
	n = argmin(LL);
	print("\nLowest per-element-loss = $(sqrt(LL[n])) for index $(n).\n")
	
	return nothing
end
main()