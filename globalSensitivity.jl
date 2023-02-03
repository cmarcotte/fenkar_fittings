include("dataManagement.jl"); 
include("plotting.jl"); 
include("fitting.jl"); 
include("sensitivity.jl");
using .dataManagement, .plotting, .fitting, .sensitivity, ProgressBars

function main()
	
	#P, lb, ub = initializeParamsAndBounds(PP);
	
	#estimateGlobalSensitivity(lb, ub; total_num_trajectory=1_000_000, num_trajectory=500_000);
	
	names= ["barone", "BR", "GP", "fit", "mf35e", "minfit"]
	inds = [1019, 929, 1301, 1302, 1306, 938]
	
	for (n,title) in ProgressBar(zip(inds, names))
		g, H, F = localSensitivity(LOSS, Float64.(PP[n]), n; title=title);
	end
	
	return nothing
end

main()