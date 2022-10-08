for i in {1..400}; do
	echo "${i}:\n"
	if [[ $((i%10)) == 0 ]]; then
		julia --project=. --threads=8 openAllFittings.jl
		echo "\n"
		cp ./fittings/tSNE.pdf ./fittings/tSNE_${i}.pdf
		#julia --project=. --threads=4 checkAllFittings.jl
	else
		julia --project=. --threads=8 fenkar_fitting.jl
		echo "\n
	fi
done