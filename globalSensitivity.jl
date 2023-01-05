const Nt = 500 # parse(Int, ARGS[1])

include("dataManagement.jl"); include("plotting.jl"); include("fitting.jl")
using .dataManagement, .plotting, .fitting

const dt = 2.0				# time between samples -- set by the data
const u0 = rand(Float64,3)
const Nsols = 62
const tspan = (0.0,Nt*dt)
const t = collect(range(tspan[1], tspan[2]; length=Nt));
const target = zeros(Float64, 1, Nt, Nsols);
const t0s = zeros(Float64, Nsols);
const BCLs = zeros(Float64, Nsols);
loadTargetData!(target, t0s, BCLs; Nt=Nt)
const LL, PP = knownFits(Nt; truncate=false);

using ForwardDiff, LinearAlgebra, PyPlot, GlobalSensitivity, ProgressBars

function main()

	pnames=["tsi", "tv1m", "tv2m", "tvp", "twm", "twp", "td", "to", "tr", "xk", "uc", "uv", "ucsi"]
	vnames = ["\$ t_0 \$", "BCL \$ \$", "\$ I_\\mathrm{stim} \$", "\$ v_0 \$", "\$ w_0 \$"]

	P, lb, ub = parametersAndBounds(Nt, Nsols, t0s, BCLs; modelfile=true, stimfile=false);
	for P in PP
		for n in 1:length(P)
			lb[n] = min(lb[n],P[n])
			ub[n] = max(ub[n],P[n])
		end
	end
	
	LOSS(x) = loss(x; u0=u0, tspan=tspan, t=t, Nt=Nt, Nsols=Nsols, target=target, M=(x)->(1.0))[1];
	
	function localSensitivity(LOSS, P, n)
		g = ForwardDiff.gradient(LOSS, P);
		H = ForwardDiff.hessian(LOSS, P);
		F = eigen(H);
		
		plt.figure(figsize=(plotting.sw,plotting.sw/2), constrained_layout=true)
		plt.hist(abs.(F.values), bins=10.0.^(-6:1:+6));
		plt.xscale("log")
		plt.xlabel("\$ \\lambda(H) \$")
		plt.ylabel("Count \$ \$")
		plt.savefig("./$(n)_hessian_eigvals.pdf", dpi=300);
		plt.close();
		
		plt.figure(figsize=(plotting.sw,plotting.sw/2), constrained_layout=true)
		plt.stem(1:13, g[1:13], label="Gradient \$ \$")
		plt.xticks(1:1:13)
		plt.grid()
		plt.legend(loc=0, edgecolor="none");
		plt.yticks([])
		plt.savefig("./$(n)_model_param_gradient.pdf", dpi=300);
		plt.close();
		
		fig, axs = plt.subplots(5, 1, figsize=(plotting.sw,plotting.sw), sharex=true, sharey=false, constrained_layout=true)
		for i in 1:5
			axs[i].stem(1:Nsols, g[(13+i):5:length(P)], label=nothing);
			axs[i].set_yticks([]);
			axs[i].grid();
			axs[i].set_xticks(1:5:Nsols)
			axs[i].set_ylabel(vnames[i])
		end
		plt.savefig("./$(n)_stim_param_gradient.pdf", dpi=300);
		plt.close();
	
		inds = sortperm(abs.(F.values));
		fig, axs = plt.subplots(10, 5, figsize=(plotting.dw,plotting.dw), sharex=true, sharey=true, constrained_layout=true)
		for i in 1:length(axs)
			axs[i].stem(1:13, abs.(F.vectors[1:13,inds[i]]), label="\$ \\lambda = $(round(F.values[inds[i]];sigdigits=2)) \$"); 
			axs[i].set_yticks([]);
			axs[i].legend(loc=0, edgecolor="none");
			axs[i].grid()
			axs[i].set_xticks(1:2:13)
		end
		plt.savefig("./$(n)_model_param_hessian_null_eigenvecs.pdf", dpi=300);
		plt.close()
	
		fig, axs = plt.subplots(5,5, figsize=(plotting.dw, plotting.dw), sharex=true, sharey="row", constrained_layout=true)
		for i in 1:5
			for j in 1:5
				tmp=F.vectors[(13+i):5:length(P), inds[j]];
				axs[i,j].stem(1:Nsols, abs.(tmp), label=nothing); 
				axs[i,j].set_yticks([]);
				#axs[i,j].legend(loc=0, edgecolor="none");
				axs[i,j].grid()
				axs[i,j].set_xticks(1:5:Nsols)
			end
			axs[i,1].set_ylabel(vnames[i])
			axs[1,i].set_title("\$ \\lambda = $(round(F.values[inds[i]]; sigdigits=2)) \$")
		end
		plt.savefig("./$(n)_stim_param_hessian_null_eigenvecs.pdf", dpi=300);
		plt.close()
		
		return nothing
	end
	
	for (n,P) in ProgressBar(enumerate(PP))
		
		#localSensitivity(LOSS, P, n);
		
		modelLoss(x) = LOSS([x; P[14:end]]);
		stimLoss(x) = LOSS([P[1:13]; x]);
		
		morris_resT = gsa(LOSS, Morris(total_num_trajectory=100000,num_trajectory=10000), collect(zip(lb,ub)));
		morris_resM = gsa(modelLoss, Morris(total_num_trajectory=100000,num_trajectory=10000), collect(zip(lb[1:13],ub[1:13])));
		morris_resS = gsa(stimLoss, Morris(total_num_trajectory=100000,num_trajectory=10000), collect(zip(lb[14:end],ub[14:end])));
		
		titles = ["Total \$ \$", "Model \$ \$", "Stimulus \$ \$"]
		fig, axs = plt.subplots(2,1,figsize=(plotting.sw,2*plotting.sw/2), sharex=true, constrained_layout=true)
		for (n,morris_res) in enumerate([morris_resT, morris_resM])
			axs[n].scatter(1:13, abs.(morris_res.means[1:13]), marker="^", color="tab:blue")
			ax2 = axs[n].twinx()
			ax2.scatter(1:13, morris_res.variances[1:13], marker="v", color="tab:red")
			axs[n].set_xticks(1:1:13)
			axs[n].set_ylabel("Morris Abs. Means \$ \$", color="tab:blue")
			ax2.set_ylabel("Morris Variances \$ \$", color="tab:red")
			axs[n].set_yscale("log")
			axs[n].set_title(titles[n])
		end
		axs[end].set_xlabel("Parameter Index \$ \$")
		plt.savefig("./$(n)_model_param_morris_gsa.pdf", dpi=300);
		plt.close()
		
		MM = length(P);
		fig, axs = plt.subplots(2,5,figsize=(5*plotting.sw,2*plotting.sw/2), sharex=true, constrained_layout=true)
		for m in 1:5
			axs[1,m].scatter(1:Nsols, abs.(morris_resT.means[(13+m):5:MM]), marker="^", color="tab:blue")
			ax2 = axs[1,m].twinx()
			ax2.scatter(1:Nsols, morris_resT.variances[(13+m):5:MM], marker="v", color="tab:red")
			axs[1,m].set_ylabel("Morris Abs. Means \$ \$", color="tab:blue")
			ax2.set_ylabel("Morris Variances \$ \$", color="tab:red")
			axs[1,m].set_yscale("log")
			axs[1,m].set_title(titles[1])
		end
		for m in 1:5
			axs[2,m].scatter(1:Nsols, abs.(morris_resT.means[m:5:MM-13]), marker="^", color="tab:blue")
			ax2 = axs[2,m].twinx()
			ax2.scatter(1:Nsols, morris_resT.variances[m:5:MM-13], marker="v", color="tab:red")
			axs[2,m].set_ylabel("Morris Abs. Means \$ \$", color="tab:blue")
			ax2.set_ylabel("Morris Variances \$ \$", color="tab:red")
			axs[2,m].set_yscale("log")
			axs[2,m].set_title(titles[3])
			axs[end,m].set_xlabel("Solution Index \$ \$ - "*vnames[m])
			axs[end,m].set_xticks(1:10:Nsols)
		end
		plt.savefig("./$(n)_stim_param_morris_gsa.pdf", dpi=300);
		plt.close()
		
	end
	
	#sobol_res = gsa(modelLoss, Sobol(order=[0,1,2]), collect(zip(lb[1:13],ub[1:13])); samples=10000)

	
	return nothing
end

main()