for i in {1..400}; do
	echo "${i}:\n"
	if [[ $((i%10)) == 0 ]]; then
		# this is the rosetta one
		/Applications/Julia-1.8.app/Contents/Resources/julia/bin/julia --project=. --threads=4 openAllFittings.jl
		echo "\n"
		#julia --project=. --threads=4 checkAllFittings.jl
	else
		# this is the native one
		julia --project=. --threads=4 fenkar_fitting.jl
		echo "\n"
	fi
done