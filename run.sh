Nt=${1}
for ((i = 1; i <= ${2}; i++)); do
	echo "${i}:\n"
	if [[ $((i%5)) == 0 ]]; then
		# this is the rosetta one
		/Applications/Julia-1.8.app/Contents/Resources/julia/bin/julia --project=. --threads=10 openAllFittings.jl ${Nt} &
		echo "\n"
		# this is the native one
		julia --project=. --threads=10 checkAllFittings.jl ${Nt} &
		echo "\n"
	else
		# this is the native one
		julia --project=. --threads=10 fenkar_fitting.jl ${Nt} 
		echo "\n"
	fi
done
