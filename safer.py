# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.7.1       #
#       Created on: 2021-06-12      #
# --------------------------------- #

"""
python3.10 safer.py -safer_config "../configs/safer/bart/safer_3_all_config.json" -output_file "../experiments/bart/safer/l_3_all_report.txt"
python3.10 safer.py -safer_config "../configs/safer/ens_train_m_noise/safer_3_all_config.json" -output_file "../experiments/snli/ens_train_m_noise_det/l_3_all_report.txt"

python3.10 safer.py -safer_config "../configs/safer/deberta/safer_test_config.json" -output_file "../experiments/snli_deberta/safer/test_report.txt"
python3.10 safer.py -safer_config "../configs/safer/deberta/safer_test_config2.json" -output_file "../experiments/snli_deberta/safer/test_report2.txt"

python3.10 safer.py -safer_config "../configs/safer/deberta/safer_3_config.json" -output_file "../experiments/snli_deberta/safer/l_3_report.txt"
python3.10 safer.py -safer_config "../configs/safer/bilstm/safer_3_config.json" -output_file "../experiments/snli_bilstm/safer/l_3_report.txt"
python3.10 safer.py -safer_config "../configs/safer/ens_train_m_noise/safer_3_config.json" -output_file "../experiments/snli/ens_train_m_noise_det/safer/l_3_report.txt"

"""

import argparse
import copy
import json
import os
import pathlib
import pickle
from random import seed as py_seed

import numpy as np
import torch

from attack_code.generate_attack_corpus import generate_attack_corpus
from attack_code.safer_certify import SaferCertify
from models.builders.module_factory import ModuleFactory



def set_seeds(SEED):

	py_seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)


def build_model(model_config):

	model = ModuleFactory.BUILD_MODULE(**model_config)
	model.to_device()
	model.eval()

	return model


if __name__ == '__main__':

	#parse command line
	parser = argparse.ArgumentParser()
	parser.add_argument("-safer_config", help="template attack config file", required=True)
	parser.add_argument("-output_file", help="template attack config file", required=True)
	parser.add_argument("-RNG_SEED", help="folder to word2vec to find pertubation set", required=False, default=1234)
	args = parser.parse_args()


	set_seeds(args.RNG_SEED)


	with open(args.safer_config, 'r') as f:
		safer_config = json.load(f)


	inference_config_file = safer_config.pop("inference_config_file")

	with open(inference_config_file, 'r') as f:
		inference_config = json.load(f)

	dataset_config = inference_config["dataset_config"]

	#remove these so we can make our own copurs/batches
	dataset_config.pop("corpus", None)
	dataset_config.pop("corpus_file", None)

	dataloader_params = inference_config["dataloader_params"]


	model_config_file = safer_config.pop("model_config_file")

	with open(model_config_file, 'r') as f:
		model_config = json.load(f)

	model = build_model(model_config)


	attack_corpus_params = safer_config.pop("attack_corpus_params")

	attack_corpus = generate_attack_corpus(**attack_corpus_params)


	synonym_json_file = safer_config.pop("synonym_json_file")

	with open(synonym_json_file, 'r') as f:
		synonyms = json.load(f)


	B_json_file = safer_config.pop("B_json_file")

	with open(B_json_file, 'r') as f:
		B = json.load(f)


	w2v_folder = safer_config.pop("w2v_folder")

	file_path = os.path.join(w2v_folder, "vocab.pkl")
	with open(file_path, "rb") as f:
		word2idx = pickle.load(f)

	file_path = os.path.join(w2v_folder, "vecs.pkl")
	with open(file_path, "rb") as f:
		vecs = pickle.load(f)


	safer = SaferCertify(
		synonyms=synonyms,
		B=B,
		word2idx=word2idx,
		vecs=vecs,
		examples=attack_corpus,						
		model=model,
		dataset_config=dataset_config, 
		dataloader_params=dataloader_params,
		**safer_config
		)

	result = safer.cert()

	output_folder = os.path.dirname(args.output_file)

	pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

	with open(args.output_file, "w") as f:
		f.write(result)






	