include("./model.jl")
using .fenkar
using DifferentialEquations
using ForwardDiff
using Optimization, OptimizationNLopt
using Random, DelimitedFiles
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
	
# make the parameters for the solution
const Nt = 500
const dt = 2.0			# time between samples -- set by the data
const tspan = (0.0,Nt*dt)
const t = collect(range(tspan[1], tspan[2]; length=Nt));
const u0 = rand(Float64,3)

const Nsols = 62
const target = zeros(Float64, 1, Nt, Nsols);
const t0s = zeros(Float64, Nsols);
const BCLs = zeros(Float64, Nsols);
try
	target .= reshape(readdlm("./data/target.txt"), size(target));
	t0s .= reshape(readdlm("./data/t0s.txt"), size(t0s));
	BCLs .= reshape(readdlm("./data/BCLs.txt"), size(BCLs));
catch
	# get the data from the recordings
	basePath = basepath = "../../Alessio_Data/2011-05-28_Rec78-103_Pace_Apex/"
	inds = getPossibleIndices(basepath)
	data = []
	m = 0
	for ind in inds, Vind in [1,2]
		m = m+1
		global t, tmp, BCL = getExpData(ind; Vind = Vind, tInds=1:Nt)
		push!(data, tmp);
		BCLs[m] = BCL;
		t0s[m] = t[findfirst(tmp[1,:] .> 0.5)]-t[1]-BCL/2; 
		# this implies Istim(t1)=(IA*sin(pi*(t1-t1-BCL/2)/BCL)^500) = IA*sin(pi/2)^500
	end

	# define target for optimization
	target[1,:,:] .= transpose(reduce(vcat, data));
	
	# write target, t0s, and BCLs to caches
	open("./data/target.txt", "w") do io
		writedlm(io, target);
	end
	open("./data/t0s.txt", "w") do io
		writedlm(io, t0s);
	end
	open("./data/BCLs.txt", "w") do io
		writedlm(io, BCLs);
	end
end

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

# model definition using (ensemble) ODE problem
function model1(θ,ensemble)
	prob = ODEProblem(fenkar!, u0, tspan, θ[1:16])
	
	function prob_func(prob, i, repeat)
		remake(prob, u0 = [target[1,1,i], θ[17+5*(i-1)], θ[18+5*(i-1)]], p = θ[[1:13; (14+5*(i-1)):(16+5*(i-1))]])
	end

	ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
	sim = solve(ensemble_prob, Tsit5(), ensemble, saveat = t, save_idxs=1:1, trajectories = 62, maxiters=Int(1e8))
	
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

function plotFits(θ,sol; target=target)

	fig, axs = plt.subplots(Int(ceil(62/8)),8, figsize=(dw,dw*1.05*(Int(ceil(62/8))/8)), 
				sharex=true, sharey=true, constrained_layout=true)
	for n in 1:Nsols
		# linear indexing into Array{Axis,2}
		axs[n].plot(t, target[1,:,n], "-k", linewidth=1.6)
		# plotting the (negated) stimulus current to make sure things line up
		axs[n].plot(t, Istim.(t,-θ[13+5*(n-1)+3],θ[13+5*(n-1)+1],θ[13+5*(n-1)+2]), "-r", linewidth=0.5)
		axs[n].plot(t, sol[1,:,n], "-C1", linewidth=1)
	end
	axs[1].set_ylim([-0.1,1.1])
	axs[1].set_xlim([0.0,1000.0])
	axs[1].set_xticks([0.0,250.0,500.0,750.0,1000.0])
	axs[1].set_xticklabels(["","250","","750",""])
	return fig, axs
end

# get known parameters and form deflation operator
const PP = knownParameters();
const LL = knownLosses();

function main()
	
	for (n,(P,L)) in enumerate(zip(PP,LL))
		# get loss and solution for parameters P
		l,sol = loss(P,SciMLBase.NullParameters()); 
		fig, axs = plotFits(P,sol; target=target)
		plt.savefig("./fittings/$n.pdf",bbox_inches="tight")
		plt.close(fig)
		
		if l > 1.05*L
			print("Oddity; parameters $n: \tL=$(L), \tl=$(l).\n")
		end
	end
	return nothing
end
main()