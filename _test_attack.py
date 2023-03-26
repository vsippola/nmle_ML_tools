# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.7.1       #
#       Created on: 2021-06-12      #
# --------------------------------- #

import json
import os
import pathlib

from attack_code.generate_attack_corpus import generate_attack_corpus
from attack_code.PWWS_attack import PWWSAttacker

from models.builders.module_factory import ModuleFactory

def build_model(model_config):

	model = ModuleFactory.BUILD_MODULE(**model_config)
	model.to_device()
	model.eval()

	return model


def main():

	attack_corpus_params = {
		"number_of_examples":1000,

		"attack_type":"hard",

		"corpus_text_file":"../parsed_data/snli_1.0/snli_1.0_test.tsv",
		
		"corpus_indexes":
		{
			"example_num":0,
			"true_label":1,
			"sentences":[2,3]
		},

		"label2int":
		{
			"entailment":0,
			"neutral":1,
			"contradiction":2
		},

		"prediction_details_file":"../experiments/snli_roberta/snli_testing_details.txt",

		"prediction_details_indexes":
		{
			"example_num":0,
			"pred_label":1,
			"probs":2,
		}		

	}

	attack_config = {
		"prob_key":"class_percentages",
		"int2label":
		{
			0:"entailment",
			1:"neutral",
			2:"contradiction"
		},
		"model_config_file":"../configs/snli_bilstm/model_attack.json",
		"inference_config_file":"../configs/snli_bilstm/testing_attack.json",
		"output_folder":"../experiments/snli_bilstm/easy_attack/"
	}

	attack_config = {
		"prob_key":"class_percentages",
		"int2label":
		{
			0:"entailment",
			1:"neutral",
			2:"contradiction"
		},
		"model_config_file":"../configs/snli_roberta/model.json",
		"inference_config_file":"../configs/snli_roberta/testing.json",
		"output_folder":"../experiments/snli_roberta/hard_attack/"
	}

	

	


	with open(attack_config["inference_config_file"], 'r') as f:
		inference_config = json.load(f)

	with open(attack_config["model_config_file"], 'r') as f:
		model_config = json.load(f)

	dataset_config = inference_config["dataset_config"]

	#remove these so we can make our own copurs/batches
	dataset_config.pop("corpus", None)
	dataset_config.pop("corpus_file", None)

	dataloader_params = inference_config["dataloader_params"]

	attack_corpus = generate_attack_corpus(**attack_corpus_params)

	model = build_model(model_config)

	attacker = PWWSAttacker(
		examples=attack_corpus, 
		model=model, 
		dataset_config=dataset_config, 
		dataloader_params=dataloader_params, 
		**attack_config)

	general_report, safe_report, attack_report = attacker.attack()

	output_folder = attack_config["output_folder"]

	pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

	for file_name, report in zip(["general.txt", "safe.txt", "attack.txt"], [general_report, safe_report, attack_report]):

		output_file = os.path.join(output_folder, file_name)

		with open(output_file, "w") as f:
			f.write(report)

	

if __name__ == '__main__':
	main()