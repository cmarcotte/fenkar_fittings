__precompile__()
module analyze

using Dierckx, GlobalSensitivity

export analyzeTraces, analyzeTrace

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
	
	# storage for APD, DI, APA for this BCL
	APD = Float64[]
	DI  = Float64[]
	APA = Float64[]
	
	if !isempty(R)
		D = derivative(Vt, R);			# V'(R)
		
		for n in 1:length(R)-1
			if D[n] > 0 && D[n+1] < 0
				push!(APD, R[n+1]-R[n])
				push!(APA, V90+maximum(Vt(R[n]:dt:R[n+1])))
			elseif D[n] < 0 && D[n+1] > 0
				push!(DI, R[n+1]-R[n])
			end
		end
	end
	return (APD, DI, APA)
end

function analyze(LL, PP, ls, sols, BCLs)
	APDs = []
	DIs  = []
	APAs = []	
	for (n,(P,L)) in enumerate(zip(PP,LL))
		# analyze sol
		tmpAPDs, tmpDIs, tmpAPAs = analyzeTraces(t, sol);
		
		for n in 1:Nsols
			for m in eachindex(tmpAPDs[n])
				push!(APDs, [BCLs[n] ;; L ;; tmpAPDs[n][m]]);
			end
			for m in eachindex(tmpDIs[n])
				push!(DIs,  [BCLs[n] ;; L ;; tmpDIs[n][m]]);
			end
			for m in eachindex(tmpAPAs[n])
				push!(APAs, [BCLs[n] ;; L ;; tmpAPAs[n][m]]);
			end
		end
				
		# plotFits(P,sol,n; target=target);
		
		if abs(l-L) > 0.05*min(L,l)
			print("Oddity: parameter set $n; \tL=$(L), \tl=$(l).\n")
		end
	end
	
	APDs = reduce(vcat, APDs);
	 DIs = reduce(vcat,  DIs);
	APAs = reduce(vcat, APAs);
	return (APDs, DIs, APAs)
end

function generate

end