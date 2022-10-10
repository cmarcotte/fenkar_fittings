include("./model.jl")
using .fenkar
using DifferentialEquations
using ForwardDiff
using Optimization, OptimizationNLopt
using Random, DelimitedFiles
using Dierckx
using Statistics

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

#using Plots
	
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

target .= reshape(readdlm("./data/target.txt"), size(target));
t0s .= reshape(readdlm("./data/t0s.txt"), size(t0s));
BCLs .= reshape(readdlm("./data/BCLs.txt"), size(BCLs));

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
	plt.savefig("./fittings/$n.pdf",bbox_inches="tight")
	plt.close(fig)
	
	#=
	plts = []
	for n in 1:Nsols
		nplt = plot();
		# linear indexing into Array{Axis,2}
		plot!(nplt, t, target[1,:,n], linecolor=:black, linewidth=4)
		# plotting the (negated) stimulus current to make sure things line up
		plot!(nplt, t, Istim.(t,-θ[13+5*(n-1)+3],θ[13+5*(n-1)+1],θ[13+5*(n-1)+2]), linecolor=:red, linewidth=1)
		plot!(nplt, t, sol[1,:,n], linecolor=:orange, linewidth=2)
		plot!(xticks=(0:250:1000,["","250","","750",""]),xlim=[0.0,1000.0],legend=false,ylim=[-0.1,1.1],size=(300,300))
		push!(plts,nplt)
	end
	comboplt = plot(plts..., link=:all, layout=(8,8), size=(1000,1000))
	=#
	
	return nothing
end

function analyzeTraces(t, Vs)
	APDs = [];
	DIs  = [];
	APAs = [];
	for n in 1:size(Vs,3)
		APD, DI, APA = analyzeTrace(t, Vs[1,:,n][:]);
		push!(APDs, APD);
		push!(DIs,   DI);
		push!(APAs, APA);
	end
	return (APDs, DIs, APAs)
end

function analyzeTrace(t, V; V90=0.2)
	dt = mean(diff(t))
	Vt = Spline1D(t[:], V.-V90, k=3);
	
	R = roots(Vt; maxn=Int(5e3));	# time points R: V(R) == V90
	D = derivative(Vt, R);			# V'(R)
	
	# storage for APD, DI, APA for this BCL
	APD = Float64[]
	DI  = Float64[]
	APA = Float64[]
	
	for n in 1:length(R)-1
		if D[n] > 0 && D[n+1] < 0
			push!(APD, R[n+1]-R[n])
			push!(APA, V90+maximum(Vt(R[n]:dt:R[n+1])))
		elseif D[n] < 0 && D[n+1] > 0
			push!(DI, R[n+1]-R[n])
		end
	end
	return (APD, DI, APA)
end

function main()
	
	# get known parameters and form deflation operator
	PP = knownParameters();
	LL = knownLosses();

	fig, axs = plt.subplots(1, 3, figsize=(dw,sw), sharex="row", sharey="col", constrained_layout=true);
	axs[1].set_xlabel("BCL [ms]")
	axs[1].set_ylabel("APD [ms]")
	axs[2].set_xlabel("BCL [ms]")
	axs[2].set_ylabel("DI  [ms]")
	axs[3].set_xlabel("BCL [ms]")
	axs[3].set_ylabel("APA [  ]")
	APDs, DIs, APAs = analyzeTraces(t, target);
	for (BCL, APD, DI, APA) in zip(BCLs, APDs, DIs, APAs)
		axs[1].plot(BCL*ones(Float64, length(APD)), APD, ".k", markersize=4)
		axs[2].plot(BCL*ones(Float64, length(DI)),   DI, ".k", markersize=4)
		axs[3].plot(BCL*ones(Float64, length(APA)), APA, ".k", markersize=4)
	end
	for (n,(P,L)) in enumerate(zip(PP,LL))
		# get loss and solution for parameters P
		l,sol = loss(P,SciMLBase.NullParameters()); 
		#=
		plotFits(P,sol; target=target);
		
		if l > 1.05*L
			print("Oddity; parameters $n: \tL=$(L), \tl=$(l).\n")
		end
		=#
		APDs, DIs, APAs = analyzeTraces(t, sol);
		for (BCL, APD, DI, APA) in zip(BCLs, APDs, DIs, APAs)
			axs[1].plot(BCL*ones(Float64, length(APD)), APD, ".C0", alpha=0.1, markersize=5)
			axs[2].plot(BCL*ones(Float64, length(DI)),   DI, ".C0", alpha=0.1, markersize=5)
			axs[3].plot(BCL*ones(Float64, length(APA)), APA, ".C0", alpha=0.1, markersize=5)
		end
		
	end
	plt.savefig("./fittings/BCLs.pdf",bbox_inches="tight")
	plt.close(fig)
	
	return nothing
end
main()