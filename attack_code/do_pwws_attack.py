# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.7.1       #
#       Created on: 2021-06-12      #
# --------------------------------- #

"""

python3.10 attack_code/do_pwws_attack.py -attack_config_file "../configs/attacks/bilstm/testing.json"

"""

import argparse
import json
import os
import pathlib
import sys

from attack_code.generate_attack_corpus import generate_attack_corpus
from attack_code.PWWS_attacker import PWWSAttacker

from models.builders.module_factory import ModuleFactory

def build_model(model_config):

	model = ModuleFactory.BUILD_MODULE(**model_config)
	model.to_device()
	model.eval()

	return model



def do_pwws(attack_config):

	pwws_attack_config = attack_config["pwws_config"]
	attack_corpus_params = attack_config["attack_corpus_params"]

	with open(pwws_attack_config["inference_config_file"], 'r') as f:
		inference_config = json.load(f)

	with open(pwws_attack_config["model_config_file"], 'r') as f:
		model_config = json.load(f)

	output_folder = pwws_attack_config["output_folder"]

	dataset_config = inference_config["dataset_config"]

	#remove these so we can make our own copurs/batches
	dataset_config.pop("corpus", None)
	dataset_config.pop("corpus_file", None)

	dataloader_params = inference_config["dataloader_params"]

	attack_corpus = generate_attack_corpus(**attack_corpus_params)

	if len(attack_corpus) == 0:
		print()
		print("No valid example found to attack")
		sys.exit()

	model = build_model(model_config)

	attacker = PWWSAttacker(
		examples=attack_corpus, 
		model=model, 
		dataset_config=dataset_config, 
		dataloader_params=dataloader_params, 
		**pwws_attack_config)

	general_report, safe_report, attack_report = attacker.attack()

	pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

	for file_name, report in zip(["general.txt", "safe.txt", "attack.txt"], [general_report, safe_report, attack_report]):

		output_file = os.path.join(output_folder, file_name)

		with open(output_file, "w") as f:
			f.write(report)

	output_file = os.path.join(output_folder, "config.json")

	with open(output_file, 'w') as f:
		json.dump(attack_config, f)
	

if __name__ == '__main__':

	#parse command line
	parser = argparse.ArgumentParser()
	parser.add_argument("-attack_config_file", help="attack confiig file", required=True)
	args = parser.parse_args()

	with open(args.attack_config_file, 'r') as f:
		attack_config = json.load(f)

	do_pwws(attack_config)