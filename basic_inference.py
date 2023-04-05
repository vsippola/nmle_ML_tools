# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.7.1       #
#       Created on: 2021-06-12      #
# --------------------------------- #


"""

python3.10 basic_inference.py -output_file "../experiments/snli/ens_train_m/snli_testing.txt" -inference_config_file "../configs/snli_ensemble_train_m/testing.json" -model_config_file "../configs/snli_ensemble_train_m/model_attack.json" -RNG_SEED 1234


python3.10 basic_inference.py -output_file "../experiments/snli_ensemble_trans_noise/snli_testing.txt" -inference_config_file "../configs/snli_ensemble_trans_noise/testing.json" -model_config_file "../configs/snli_ensemble_trans_noise/model.json" -RNG_SEED 1234


python3.10 basic_inference.py -output_file "../experiments/snli_ensemble_trans_vote/snli_testing.txt" -inference_config_file "../configs/snli_ensemble_trans_vote/testing.json" -model_config_file "../configs/snli_ensemble_trans_vote/model.json" -RNG_SEED 1234

python3.10 basic_inference.py -output_file "../experiments/snli_ensemble_vote/snli_testing.txt" -inference_config_file "../configs/snli_ensemble_vote/testing.json" -model_config_file "../configs/snli_ensemble_vote/model.json" -RNG_SEED 1234

python3.10 basic_inference.py -output_file "../experiments/snli_ensemble_trans/snli_testing.txt" -inference_config_file "../configs/snli_ensemble_trans/testing.json" -model_config_file "../configs/snli_ensemble_trans/model.json" -RNG_SEED 1234

python3.10 basic_inference.py -output_file "../experiments/snli_ensemble/snli_testing.txt" -inference_config_file "../configs/snli_ensemble/testing.json" -model_config_file "../configs/snli_ensemble/model.json" -RNG_SEED 1234


python3.10 basic_inference.py -output_file "../experiments/snli_bilstm/snli_testing.txt" -inference_config_file "../configs/snli_bilstm/testing.json" -model_config_file "../configs/snli_bilstm/model_testing.json" -RNG_SEED 1234

python3.10 basic_inference.py -output_file "../experiments/snli_roberta/snli_testing.txt" -inference_config_file "../configs/snli_roberta/testing.json" -model_config_file "../configs/snli_roberta/model.json" -RNG_SEED 1234

python3.10 basic_inference.py -output_file "../experiments/snli_deberta/snli_testing.txt" -inference_config_file "../configs/snli_deberta/testing.json" -model_config_file "../configs/snli_deberta/model.json" -RNG_SEED 1234

python3.10 basic_inference.py -output_file "../experiments/snli_bart/snli_testing.txt" -inference_config_file "../configs/snli_bart/testing.json" -model_config_file "../configs/snli_bart/model.json" -RNG_SEED 1234

"""

import argparse
import json
import os
import pathlib

import torch

from models.builders.module_factory import ModuleFactory
from training.inference_evaluator_builder import InferenceEvaluatorBuilder



def build_model(model_config):

	model = ModuleFactory.BUILD_MODULE(**model_config)
	model.to_device()

	return model


def build_inference_evaluators(inference_config):

	inference_evalutaor_builder = InferenceEvaluatorBuilder()
	inference_evalutaor_builder.configure(**inference_config)
	inference_evaluator = inference_evalutaor_builder.build()

	return inference_evaluator


def main():

	#parse command line
	parser = argparse.ArgumentParser()
	parser.add_argument("-output_file", help="folder to store inference results", required=True)
	parser.add_argument("-inference_config_file", help="config for the ingerece evaluator", required=True)
	parser.add_argument("-model_config_file", help="config for the ingerece evaluator", required=True)
	parser.add_argument("-RNG_SEED", help="RNG seed value", default=None, required=False, type=int)
	args = parser.parse_args()


	with open(args.inference_config_file, 'r') as f:
		inference_config = json.load(f)

	with open(args.model_config_file, 'r') as f:
		model_config = json.load(f)

	if args.RNG_SEED is not None:
		torch.manual_seed(args.RNG_SEED)

	inference_evaluator = build_inference_evaluators(inference_config)

	model = build_model(model_config)

	results = inference_evaluator(model)

	output_folder = os.path.dirname(args.output_file)

	pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

	print(results)

	with open(args.output_file, "w") as f:
		f.write(f"{results}\n")		
	

if __name__ == '__main__':
	main()