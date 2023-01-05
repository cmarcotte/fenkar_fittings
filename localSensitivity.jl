LOSS(P) = loss(P,SciMLBase.NullParameters())[1]
g = ForwardDiff.gradient(LOSS, P);
H = ForwardDiff.hessian(LOSS, P);
F = eigen(H);

plt.figure(figsize=(sw,sw/2), constrained_layout=true)
plt.stem(1:13, g[1:13], label="Gradient \$ \$")
plt.xticks([1,5,10,13])
plt.grid()
plt.legend(loc=0, edgecolor="none");
plt.yticks([])
plt.savefig("./barone_model_param_gradient.pdf", dpi=300);
plt.close();

vnames = ["\$ t_0 \$", "BCL \$ \$", "\$ I_\\mathrm{stim} \$", "\$ v_0 \$", "\$ w_0 \$"]
fig, axs = plt.subplots(5, 1, figsize=(sw,sw), sharex=true, sharey=false, constrained_layout=true)
for i in 1:5
	axs[i].stem(1:Nsols, g[(13+i):5:length(P)], label=nothing);
	axs[i].set_yticks([]);
	axs[i].grid();
	axs[i].set_xticks(1:5:Nsols)
	axs[i].set_ylabel(vnames[i])
end
plt.savefig("./barone_stim_param_gradient.pdf", dpi=300);
plt.close();

#=
	This plots the 50 smallest eigenmodes over the model parameter indices; eigenmodes where most of the energy is in a single index and the eigenvalue is small indicate that large perturbations in that model parameter have minimal impact on the loss -- i.e., insensitivity.
=#
inds = sortperm(abs.(F.values));
fig, axs = plt.subplots(10, 5, figsize=(,10), sharex=true, sharey=true, constrained_layout=true)
for i in 1:length(axs)
	axs[i].stem(1:13, abs.(F.vectors[1:13,inds[i]]), label="\$ \\lambda = $(round(F.values[inds[i]];sigdigits=2)) \$"); 
	axs[i].set_yticks([]);
	axs[i].legend(loc=0, edgecolor="none");
	axs[i].grid()
	axs[i].set_xticks([1,5,10,13])
end
plt.savefig("./barone_model_param_hessian_null_eigenvecs.pdf", dpi=300);
plt.close()

#=
	This plots the accumulated marginal eigenmodes, split across the stimulus indices?
=#
fig, axs = plt.subplots(5,5, figsize=(10,10), sharex=true, sharey="row", constrained_layout=true)
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
plt.savefig("./barone_stim_param_hessian_null_eigenvecs.pdf", dpi=300);
plt.close()
