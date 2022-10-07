for i in {1..1000}; do
	echo "${i}:\n"
	julia --project=. --threads=4 fenkar_fitting.jl
	echo "\n"
	if [[ $((i%10)) == 0 ]]; then
		julia --project=. --threads=4 openAllFittings.jl
		echo "\n"
		cp ./fittings/tSNE.pdf ./fittings/tSNE_${i}.pdf
		#julia --project=. --threads=4 checkAllFittings.jl
	fi
done