.PHONY: run run-beta run-vanilla

run:
	python run.py -c configs/afhq_experiments/cvae_afhq.yaml

run-beta:
	python run.py -c configs/afhq_experiments/betavae_afhq.yaml

run-vanilla:
	python run.py -c configs/afhq_experiments/vanillavae_afhq.yaml