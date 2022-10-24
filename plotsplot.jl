using Plots

function plotFits(θ,sol; target=target)

	plts = []
	for n in 1:Nsols
		# linear indexing into Array{Axis,2}
		plt = plot(t, target[1,:,n], color=:black, linewidth=1.6)
		plot!(plt, t, Istim.(t,-θ[13+5*(n-1)+3],θ[13+5*(n-1)+1],θ[13+5*(n-1)+2]), linewidth=0.5, color=:red)
		plot!(plt, t, sol1[1,:,n], linewidth=1, color=:blue)
		plot!(plt, t, sol[1,:,n], linewidth=1, color=:orange)
		plot!(plt, legend=false, size=(200,200), xlim=[t[begin],t[end]],ylim=[-0.1,1.1])
		push!(plts, plt);
	end
	return plot(plts..., size=(1800,1800), layout=(8,8), link=:all)
end