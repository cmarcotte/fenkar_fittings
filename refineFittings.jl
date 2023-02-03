const Nt = 500

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

function pinExceptSetParams!(P, lb, ub, ns::T, n) where T <: Integer
	m = 13 .+ (ns-1)*5 .+ (1:5);
	for mm in 14:length(P)
		if !(mm in m)
			P[ns]  = PP[n][ns];
			lb[ns] = PP[n][ns];
			ub[ns] = PP[n][ns];
		end
	end
	return nothing
end

function pinExceptSetParams!(P, lb, ub, inds::AbstractArray{T}, n) where T <: Integer
	for ns in inds
		pinExceptSetParams!(P, lb, ub, ns, n)	
	end
	return nothing
end

function refineFitParams(inds)
	
	for n in inds
		
		P, lb, ub = parametersAndBounds(Nt, Nsols, t0s, BCLs);
		P[1:13] .= PP[n][1:13];
		fitting.pinModelParams!(P, lb, ub; tol=0.0);
		
		# fix everything except the one set of init/stim params
		ns = 61:62#rand(1:Nsols,2); 
		#pinExceptSetParams!(P, lb, ub, ns, n);	
			
		l0, sol0 = loss(PP[n]; tspan=tspan, t=t, target=target, Nt=Nt, Nsols=Nsols, M=(x)->1.0);
		result, l, sol = optimizeParams(P, lb, ub; u0=u0, tspan=tspan, t=t, Nt=Nt, Nsols=Nsols, target=target, M=(x)->1.0);
		
		print("="^80)
		print("\n\tOriginal loss is $(l0), new loss is $(l).\n")
		print("Model params preserved? $(all(result.u[1:13] .≈ P[1:13])).\n");
		print("Loss decreased? $(l < l0).\n");
		if l < l0 && all(result.u[1:13] .≈ P[1:13])
			writeParams(result.u, l, "./fittings/Nt_$(Nt)/$(n).txt");
			LL[n] = l;
			PP[n][14:end] .= result.u[14:end];
			print("\tn=$(n) is updated.\n")
		else
			print("\tn=$(n) is not updated.\n")
		end
		print("="^80)
	end
	return nothing
end

function main()
	#names= ["Barone 𝑒𝑡 𝑎𝑙", "BR", "GP", "fit"]#, "mf35", ""mf35e"]
	inds = [1019, 1307, 1301, 938]#, 1302, 1306]
	for _ in 1:5
		refineFitParams(1:1307);
	end
	return nothing
end

main()