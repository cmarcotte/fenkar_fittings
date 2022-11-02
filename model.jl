module fenkar

	export fenkar!, noise!, Istim

	# model stuff
	function H(x;k=100.0)
		return 0.5*(1.0 + tanh(k*x))
	end
	#=
	function hat(x;width=1.0)
		if  abs(x) <= width/2.0
			return 1.0
		else
			return 0.0
		end
	end
	=#
	function Monophasic(t,IA,t0,TI)
		return IA*H(t-t0;k=1.0)*sin(pi*(t-t0)/TI)^500
	end
	
	function Biphasic(t,IA,t0,TI)
		return IA*H(t-t0;k=1.0)*500*(pi/TI)*cos(pi*(t-t0)/TI)*sin(pi*(t-t0)/TI)^499
	end
	#=
	function MonoHat(t,IA,t0,TI; width=5.0)
		return IA*H(t-t0;k=1.0)*hat((t-t0-TI/2)%TI;width=width)
	end
	
	function BiHat(t,IA,t0,TI; width=5.0)
		return MonoHat(t,IA,t0,TI; width=width)*sign((t-t0-TI/2)%TI)
	end
	=#
	
	Istim(t,IA,t0,TI) = Monophasic(t,IA,t0,TI);
	
	function fenkar!(dx, x, p, t)
		
		# parameters
		@views tsi,tv1m,tv2m,tvp,twm,twp,td,to,tr,xk,uc,uv,ucsi,t0,TI,IA = p[1:16]
		
		# fenkar dynamics
		dx[1] = Istim(t,IA,t0,TI) - (x[1]*H(uc-x[1])/to + H(x[1]-uc)/tr - x[2]*H(x[1]-uc)*(1.0-x[1])*(x[1]-uc)/td - x[3]*H(x[1]-ucsi;k=xk)/tsi)
		dx[2] = H(uc-x[1])*(1.0-x[2])/(tv1m*H(uv-x[1]) + tv2m*H(x[1]-uv)) - H(x[1]-uc)*x[2]/tvp
		dx[3] = H(uc-x[1])*(1.0-x[3])/twm - H(x[1]-uc)*x[3]/twp
		
		return nothing
	end
		
	function noise!(dx, x, p, t)
		dx[1] = 0.05*exp(-50.0*x[1]) + 0.10*exp(-50.0*abs(x[1]-1.0))
		return nothing
	end

end
