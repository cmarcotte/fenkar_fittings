include("dataManagement.jl"); 
include("fitting.jl"); 
include("sensitivity.jl");
using .dataManagement, .fitting, .sensitivity, ProgressBars, CairoMakie, MathTeXEngine

const sw = 3.40457
const dw = 7.05826
const kw = 80

function param_fits(t, data, fits, inds, names, filename)
	fig = Figure(resolution = (sw*kw, sw*kw), figure_padding = 10, fontsize = 10, font = texfont()) #fonts = (; regular = texfont(), bold = texfont()))
	ga = fig[1,1] = GridLayout()
	
	axs = [ Axis(ga[1,1], ylabel="EPI"), Axis(ga[2,1], xlabel=L"$ t $ [ms]", ylabel="ENDO")]	
	
	for i in 1:2
		lines!(axs[i], t, data[:,i], label="Exp. Data", color=:black, linewidth=2)
		for j in eachindex(fits)
			lines!(axs[i], t, fits[j][:,i], label="$(names[j])", linewidth=1)
		end
	end
	
	# legend
	if length(fits) < 6
		Legend(ga[0,:], axs[1], orientation=:horizontal, tellwidth=false, tellheight=true, framevisible=false, margin=(-8,-8,-8,-8), padding=0, valign=:bottom, nbanks=2)	
	end
	
	# subplot labels
	Label(ga[1,1,TopLeft()], "(a)", padding=(0,30,20,0), halign=:left, tellheight=false)
	Label(ga[2,1,TopLeft()], "(b)", padding=(0,30,20,0), halign=:left, tellheight=false)
	
	# link axes
	linkaxes!(axs[1], axs[2])

	# hide some frame decorations
	hidedecorations!(axs[1], grid=true, ticks=false, ticklabels=false, label=false)
	hidexdecorations!(axs[1], ticks=false, ticklabels=true)
	hidedecorations!(axs[2], grid=true, ticks=false, ticklabels=false, label=false)
	
	# control spacing
	colgap!(ga, 4)	
	rowgap!(ga, 8)	
	
	# explicit limits?
	xlims!.((axs[1],axs[2]), 0,1000)
	ylims!.((axs[1],axs[2]), 0,1)

	# hmmm
	resize_to_layout!(fig)

	# save
	save(filename, fig, pt_per_unit=2)
	
	return nothing
end

function form_fits(inds)

	fits = []
	losses = []
	for n in ProgressBar(inds)
		l, ens_sol = loss(PP[n]; u0=sensitivity.u0, tspan=sensitivity.tspan, t=sensitivity.t, Nt=sensitivity.Nt, Nsols=sensitivity.Nsols, target=sensitivity.target, M=(x)->(1.0))
		push!(fits, ens_sol[1,:,end-1:end]);
		push!(losses, l);
	end
	
	return fits, losses
end

function partialLoss(sols, data)
	return sqrt(sum(abs2,sols.-data)/size(sols,1))
end
function partial_fits(data)
	partialfits = []
	for (n,P) in ProgressBar(enumerate(PP))
		l, ens_sol = loss(P; u0=sensitivity.u0, tspan=sensitivity.tspan, t=sensitivity.t, Nt=sensitivity.Nt, Nsols=sensitivity.Nsols, target=sensitivity.target, M=(x)->(1.0));
		push!(partialfits, partialLoss(ens_sol[1,:,end-1:end], data));
	end
	inds = partialsortperm(partialfits, 1:5);
	return inds
end

function main()
	
	t = sensitivity.t;
	data = sensitivity.target[1,:,end-1:end];
	
	function partial_loss(sol)
		return sum(abs2,sols.-data[1,:,end-1:end])/prod(size(sols))
	end
	
	names= ["Barone 𝑒𝑡 𝑎𝑙", "BR", "GP", "fit"]#, "mf35", ""mf35e"]
	inds = [1019, 1307, 1301, 938]#, 1302, 1306]
	
	fits, losses = form_fits(inds)
	param_fits(t, data, fits, inds, names, "./param_fits.pdf")
	print("Losses = $(losses)\n");
	
	inds = partial_fits(data);
	fits, losses = form_fits(inds)
	param_fits(t, data, fits, inds, inds, "./param_fits2.pdf")
	print("Losses = $(losses)\n");
	
	inds = partialsortperm(LL, 1:5);
	fits, losses = form_fits(inds)
	param_fits(t, data, fits, inds, inds, "./param_fits3.pdf")
	print("Losses = $(losses)\n");
	
	return nothing
end

main()