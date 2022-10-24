using DelimitedFiles, GlobalSensitivity, QuasiMonteCarlo, Statistics, OrdinaryDiffEq, Plots
#=
const Nt = 1200;

function knownParametersLosses()
	LL = []
	PP = []
	loading = true
	ind = 0
	while loading
		try
			ind = ind + 1
			tmp = readdlm("./fittings/Nt_$(Nt)/$(ind).txt"; comments=true, comment_char='#');
			push!(PP, vcat(tmp[1,1:13][:],transpose(tmp[2:end,1:5])[:]))
			tmp = readdlm("./fittings/Nt_$(Nt)/$(ind).txt");
			push!(LL, Float64(tmp[1,4]))
		catch
			loading = false
		end
	end
	print("There are $(length(PP)) known model parameter sets!\n");
	return LL,PP
end

function 

function main()
	LL, PP = knownParameters();
	
	data = reduce(hcat, PP);
	data = Float64.(data);
		
	
		
	return nothing
end

main()
=#

function f(du,u,p,t)
  du[1] = p[1]*u[1] - p[2]*u[1]*u[2] #prey
  du[2] = -p[3]*u[2] + p[4]*u[1]*u[2] #predator
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(f,u0,tspan,p)
t = collect(range(0, stop=10, length=200))

f1 = function (p)
  prob_func(prob,i,repeat) = remake(prob;p=p[:,i])
  ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
  sol = solve(ensemble_prob,Tsit5(),EnsembleThreads();saveat=t,trajectories=size(p,2))
  # Now sol[i] is the solution for the ith set of parameters
  out = zeros(2,size(p,2))
  for i in 1:size(p,2)
    out[1,i] = mean(sol[i][1,:])
    out[2,i] = maximum(sol[i][2,:])
  end
  out
end

samples = 5000
lb = [1.0, 1.0, 1.0, 1.0]
ub = [5.0, 5.0, 5.0, 5.0]
sampler = SobolSample()
A,B = QuasiMonteCarlo.generate_design_matrices(samples,lb,ub,sampler)

sobol_result = gsa(f1,Sobol(),A,B,batch=true)

p1 = bar(["a","b","c","d"],sobol_result.ST[1,:],title="Total Order Indices prey",legend=false)
p2 = bar(["a","b","c","d"],sobol_result.S1[1,:],title="First Order Indices prey",legend=false)
p1_ = bar(["a","b","c","d"],sobol_result.ST[2,:],title="Total Order Indices predator",legend=false)
p2_ = bar(["a","b","c","d"],sobol_result.S1[2,:],title="First Order Indices predator",legend=false)
plot(p1,p2,p1_,p2_)

