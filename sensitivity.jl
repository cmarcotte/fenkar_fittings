__precompile__()
module sensitivity
include("dataManagement.jl"); include("plotting.jl"); include("fitting.jl")
using .dataManagement, .plotting, .fitting
using ForwardDiff, LinearAlgebra, PyPlot, GlobalSensitivity, ProgressBars, JLD2, FileIO
using JLD2

const Nt = 500
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

export Nt, Nsols, LL, PP
export LOSS, initializeParamsAndBounds, localSensitivity, estimateGlobalSensitivity, estimateGlobalSensitivities

pnames=["tsi", "tv1m", "tv2m", "tvp", "twm", "twp", "td", "to", "tr", "xk", "uc", "uv", "ucsi"]
vnames = ["\$ t_0 \$", "BCL \$ \$", "\$ I_\\mathrm{stim} \$", "\$ v_0 \$", "\$ w_0 \$"]

LOSS(x) = loss(x; u0=u0, tspan=tspan, t=t, Nt=Nt, Nsols=Nsols, target=target, M=(x)->(1.0))[1];

function initializeParamsAndBounds(PP; modelfile=false, stimfile=false)
	P, lb, ub = parametersAndBounds(Nt, Nsols, t0s, BCLs; modelfile=modelfile, stimfile=stimfile);
	for P in PP
		for n in 1:length(P)
			lb[n] = min(lb[n],P[n])
			ub[n] = max(ub[n],P[n])
		end
	end
	return (P, lb, ub)
end

function localSensitivity(LOSS, P, n; title=nothing)
	g = ForwardDiff.gradient(LOSS, P);
	H = ForwardDiff.hessian(LOSS, P);
	F = eigen(H);
	
	plt.figure(figsize=(plotting.sw,plotting.sw/2), constrained_layout=true)
	plt.hist(abs.(F.values), bins=10.0.^(-6:1:+6));
	plt.xscale("log")
	plt.xlabel("\$ \\lambda(H) \$")
	plt.ylabel("Count \$ \$")
	plt.title("$(title) \$ \$")
	plt.savefig("./$(n)_hessian_eigvals.pdf", dpi=300);
	plt.close();
	
	plt.figure(figsize=(plotting.sw,plotting.sw/2), constrained_layout=true)
	plt.stem(1:13, g[1:13], label="Gradient \$ \$")
	plt.xticks(1:1:13)
	plt.grid()
	plt.legend(loc=0, edgecolor="none");
	plt.yticks([])
	plt.title("$(title) \$ \$")
	plt.savefig("./$(n)_model_param_gradient.pdf", dpi=300);
	plt.close();
	
	fig, axs = plt.subplots(5, 1, figsize=(plotting.sw,plotting.sw), sharex=true, sharey=false, constrained_layout=true)
	for i in 1:5
		axs[i].stem(1:Nsols, g[(13+i):5:length(P)], label=nothing);
		axs[i].set_yticks([]);
		axs[i].grid();
		axs[i].set_xticks(1:10:Nsols)
		axs[i].set_ylabel(vnames[i])
	end
	plt.savefig("./$(n)_stim_param_gradient.pdf", dpi=300);
	plt.close();

	inds = sortperm(abs.(F.values));
	fig, axs = plt.subplots(10, 5, figsize=(plotting.dw,plotting.dw), sharex=true, sharey=true, constrained_layout=true)
	for i in 1:length(axs)
		axs[i].stem(1:13, abs.(F.vectors[1:13,inds[i]]), label="\$ \\lambda = $(round(real(F.values[inds[i]]);sigdigits=2)) \$"); 
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
			axs[i,j].set_xticks(1:10:Nsols)
		end
		axs[i,1].set_ylabel(vnames[i])
		axs[1,i].set_title("\$ \\lambda = $(round(real(F.values[inds[i]]); sigdigits=2)) \$")
	end
	plt.savefig("./$(n)_stim_param_hessian_null_eigenvecs.pdf", dpi=300);
	plt.close()
	
	return (g,H,F)
end

#=
function plotGlobalSensitivity(morris_resT, morris_resM, morris_resS, n; MM=323)

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
		if m==1;
			axs[1,m].set_ylabel("Morris Abs. Means \$ \$", color="tab:blue");
		elseif m==5;
			ax2.set_ylabel("Morris Variances \$ \$", color="tab:red")
		end
		axs[1,m].set_yscale("log")
		axs[1,m].set_title(titles[1])
	end
	for m in 1:5
		axs[2,m].scatter(1:Nsols, abs.(morris_resT.means[m:5:MM-13]), marker="^", color="tab:blue")
		ax2 = axs[2,m].twinx()
		ax2.scatter(1:Nsols, morris_resT.variances[m:5:MM-13], marker="v", color="tab:red")
		if m==1;
			axs[2,m].set_ylabel("Morris Abs. Means \$ \$", color="tab:blue")
		elseif m==5;
			ax2.set_ylabel("Morris Variances \$ \$", color="tab:red")
		end
		axs[2,m].set_yscale("log")
		axs[2,m].set_title(titles[3])
		axs[end,m].set_xlabel("Solution Index \$ \$ - "*vnames[m])
		axs[end,m].set_xticks(1:10:Nsols)
	end
	plt.savefig("./$(n)_stim_param_morris_gsa.pdf", dpi=300);
	plt.close()
	
	return nothing

end

function estimateGlobalSensitivity(P, lb, ub, n; total_num_trajectory=100000, num_trajectory=10000)

	modelLoss(x) = LOSS([x; P[14:end]]);
	stimLoss(x) = LOSS([P[1:13]; x]);
	
	morris_resT = gsa(LOSS, Morris(total_num_trajectory=total_num_trajectory,num_trajectory=num_trajectory), collect(zip(lb,ub)));
	morris_resM = gsa(modelLoss, Morris(total_num_trajectory=total_num_trajectory,num_trajectory=num_trajectory), collect(zip(lb[1:13],ub[1:13])));
	morris_resS = gsa(stimLoss, Morris(total_num_trajectory=total_num_trajectory,num_trajectory=num_trajectory), collect(zip(lb[14:end],ub[14:end])));
	
	#plotGlobalSensitivity(morris_resT, morris_resM, morris_resS, n);
	save("./$(n)_morris.jld2", Dict("morris_resT"=>morris_resT, "morris_resM"=>morris_resM, "morris_resS"=>morris_resS));
	
	return nothing
	
end

function estimateGlobalSensitivities(PP, lb, ub; total_num_trajectory=100000, num_trajectory=10000)
	for (n,P) in ProgressBar(enumerate(PP))
		estimateGlobalSensitivity(P, lb, ub, n; total_num_trajectory=total_num_trajectory, num_trajectory=num_trajectory);
	end
	return nothing
end
=#

function estimateGlobalSensitivity(lb, ub; total_num_trajectory=100000, num_trajectory=10000, MM=323)
	
	#try
		morris_res = load("./morris.jld2", "morris_res");
		sobol_res = load("./sobol.jld2", "sobol_res");
	#catch
	#	morris_res = gsa(LOSS, Morris(total_num_trajectory=total_num_trajectory,num_trajectory=num_trajectory), collect(zip(lb,ub)));	
	#	save("./morris.jld2", Dict("morris_res"=>morris_res));
	#end
	
	# plot morris results
	titles = ["Total \$ \$", "Model \$ \$", "Stimulus \$ \$"]
	pticks=["\$ \\tau_{si} \$", "\$ \\tau_{v1}^{-} \$", "\$ \\tau_{v2}^{-} \$", "\$ \\tau_{v}^{+} \$", "\$ \\tau_{w}^{-} \$", "\$ \\tau_{w}^{+} \$", "\$ \\tau_{d} \$", "\$ \\tau_{o} \$", "\$ \\tau_{r} \$", "\$ k \$", "\$ u_c \$", "\$ u_v \$", "\$ u_{c}^{si} \$"]
	fig, axs = plt.subplots(1,1,figsize=(plotting.sw,plotting.sw/2), sharex=true, constrained_layout=true, squeeze=false)
	for (n,morris_res) in enumerate([morris_res])
		axs[n].scatter(1:13, abs.(morris_res.means[1:13]), marker="^", color="tab:blue")
		ax2 = axs[n].twinx()
		axs[n].tick_params(axis="y", colors="tab:blue", which="both")
		ax2.tick_params(axis="y", colors="tab:red", which="both")
		ax2.scatter(1:13, morris_res.variances[1:13], marker="v", color="tab:red")
		axs[n].set_ylabel("Morris Abs. Means \$ \$", color="tab:blue")
		ax2.set_ylabel("Morris Variances \$ \$", color="tab:red")
		axs[n].set_yscale("log")
		ax2.set_yscale("log")
	end
	axs[end].set_xlabel("Parameter \$ \$")
	axs[end].set_xticks(1:length(pticks), pticks)
	plt.savefig("./model_param_morris_gsa.pdf", dpi=300);
	plt.close()
	
	fig, axs = plt.subplots(5,1,figsize=(plotting.sw,5*plotting.sw/2+1), sharex=true, constrained_layout=true, squeeze=false)
	for m in 1:5
		axs[m].scatter(1:Nsols, abs.(morris_res.means[(13+m):5:MM]), marker="^", color="tab:blue")
		ax2 = axs[m].twinx()
		axs[m].tick_params(axis="y", colors="tab:blue", which="both")
		ax2.tick_params(axis="y", colors="tab:red", which="both")
		ax2.scatter(1:Nsols, morris_res.variances[(13+m):5:MM], marker="v", color="tab:red")
		axs[m].set_ylabel("Morris Abs. Means \$ \$", color="tab:blue");
		ax2.set_ylabel("Morris Variances \$ \$", color="tab:red")
		axs[m].set_yscale("log")
		ax2.set_yscale("log")
		axs[m].set_title("$(vnames[m])")
	end
	axs[end].set_xlabel("Solution Index \$ \$")
	axs[end].set_xlim([1,Nsols])
	axs[end].set_xticks(1:10:Nsols)
	plt.savefig("./stim_param_morris_gsa.pdf", dpi=300);
	plt.close()
	
	# plot sobol results
	
	return nothing
	
end

end