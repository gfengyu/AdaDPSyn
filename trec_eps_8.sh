echo "AdaDPSyn on TREC dataset, eps = 8"
for seed in {0..4}; do
	python run_trec.py --sigma_radius 15 --sigma_test 5 --sigma_avg 1.09 --T_update 2 --llambda 0.25 --sample_seed $seed --noise_seed $seed
done