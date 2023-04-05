# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.7.1       #
#       Created on: 2023-03-12      #
# --------------------------------- #

"""

python3.10 training.py -working_folder "../experiments/snli_bilstm_4" -config_folder "../configs/snli_bilstm/" -RNG_SEED 1234

python3.10 training.py -working_folder "../experiments/snli/ens_train_m" -config_folder "../configs/snli_ensemble_train_m/" -RNG_SEED 1234
python3.10 training.py -working_folder "../experiments/snli/ens_train_m_noise" -config_folder "../configs/snli_ensemble_train_m_noise/" -RNG_SEED 1234
python3.10 training.py -working_folder "../experiments/snli/ens_train_m_model" -config_folder "../configs/snli_ensemble_train_m_model/" -RNG_SEED 1234

python3.10 training.py -working_folder "../experiments/snli/ens_train_mlp" -config_folder "../configs/snli_ensemble_train_mlp/" -RNG_SEED 1234

python3.10 training.py -working_folder "../experiments/snli/ens_train_m2" -config_folder "../configs/snli_ensemble_train_m2/" -RNG_SEED 1234

"""

import argparse
import json
import os
import pathlib

import torch

from models.builders.module_factory import ModuleFactory
from training.model_trainer_builder import ModelTrainerBuilder



def build_model(model_config):

	model = ModuleFactory.BUILD_MODULE(**model_config)
	model.to_device()

	return model


def build_model_trainer(model_trainer_config):

	model_trainer_builder = ModelTrainerBuilder()
	model_trainer_builder.configure(**model_trainer_config)
	model_trainer = model_trainer_builder.build()

	return model_trainer


def main():

	#parse command line
	parser = argparse.ArgumentParser()
	parser.add_argument("-working_folder", help="folder to store checkpoints and logs", required=True)
	parser.add_argument("-config_folder", help="folder containing model and trianing configs", required=True)
	parser.add_argument("-RNG_SEED", help="RNG seed valui", default=None, required=False, type=int)
	args = parser.parse_args()

	model_config_fname = os.path.join(args.config_folder, "model.json")

	with open(model_config_fname, 'r') as f:
		model_config = json.load(f)

	model_trainer_config_fname = os.path.join(args.config_folder, "training.json")

	with open(model_trainer_config_fname, 'r') as f:
		model_trainer_config = json.load(f)

	pathlib.Path(args.working_folder).mkdir(parents=True, exist_ok=True)

	config_folder = os.path.join(args.working_folder, "configs/")

	pathlib.Path(config_folder).mkdir(parents=True, exist_ok=True)	

	for config, fname in zip([model_config, model_trainer_config],["model.json", "training.json"]):

		output_filename = os.path.join(config_folder, fname)
		json_object = json.dumps(config, indent=4)
		with open(output_filename, 'w') as f:
			f.write(json_object)

	model_trainer_config["checkpoint_folder"] = args.working_folder
	model_trainer_config["display_config"]["log_file"] = os.path.join(args.working_folder, "training_log.txt")
	model_trainer_config["display_config"]["summary_file"] = os.path.join(args.working_folder, "summary_test.txt")

	if args.RNG_SEED is not None:
		torch.manual_seed(args.RNG_SEED)	

	#build model and trainer
	model = build_model(model_config)

	model_trainer = build_model_trainer(model_trainer_config)
	
	#run training
	model_trainer(model)

	for name, param in model.named_parameters():
	    print (name, param.data)


if __name__ == '__main__':

	main()