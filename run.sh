Nt=${1}
threads=8
for ((i = 1; i <= ${2}; i++)); do
	echo "${i}:\n"
	if [[ $((i%10)) == 0 ]] || [[ ${i} == ${2} ]]; then
		# this is the rosetta one
		/Applications/Julia-1.8.app/Contents/Resources/julia/bin/julia --project=. --threads=${threads} projectFittings.jl ${Nt} &
		echo "\n"
		# this is the native one
		julia --project=. --threads=${threads} checkAllFittings.jl ${Nt} &
		julia --project=. --threads=${threads} rewriteFittings.jl ${Nt} &
		echo "\n"
	else
		# this is the native one
		julia --project=. --threads=${threads} fenkar_fitting.jl ${Nt} 
		echo "\n"
	fi
done
