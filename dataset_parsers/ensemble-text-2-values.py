'''

python3.10 ensemble-text-2-values.py -loading_file ./dataloading_dev.json -model_file ./model.json -output_file ../../parsed_data/ens/ens_values_dev.pkl
python3.10 ensemble-text-2-values.py -loading_file ./dataloading_train.json -model_file ./model.json -output_file ../../parsed_data/ens/ens_values_train.pkl

'''

import argparse
import json
import os
import pathlib
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from dataloaders.dataset_factory import DatasetFactory
from models.builders.module_factory import ModuleFactory


def make_dataloader(loading_configs):

	dataset = DatasetFactory.BUILD_DATASET(**loading_configs["dataset_config"])

	loading_configs["dataloader_params"]["collate_fn"] = dataset.collate_fn
	dataloader = DataLoader(dataset, **loading_configs["dataloader_params"])

	return dataloader

def make_model(model_config):

	model = ModuleFactory.BUILD_MODULE(**model_config)
	model.to_device()

	return model


def calculate_values(model, dataloader):

	ensemble_values_corpus = []

	model.eval()

	with torch.no_grad():

		ensemble_values_corpus = []

		for batch in dataloader:

			state_object = model(batch)

			number = state_object["example_numbers"]
			true_labels = state_object["true_labels"]
			ensemble_values = state_object["ensemble_values"]

			for num, label, values in zip(number, true_labels.cpu().numpy(), ensemble_values.cpu()):

				example = [num, label, values]

				ensemble_values_corpus.append(example)

	return ensemble_values_corpus


def save_corpus(corpus, file):

	#create output folder if it doesn't exist
	output_dir = os.path.dirname(file)
	
	if not (os.path.isdir(output_dir)):
		print()
		print(f'folder {output_dir} does not exist creating it')
		pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

	with open(file, "wb") as f:
		pickle.dump(corpus, f)


def main():	
	parser = argparse.ArgumentParser()
	parser.add_argument("-loading_file", help="dataset/loader configs", required=True)
	parser.add_argument("-model_file", help="ensemble values model config file", required=True)
	parser.add_argument("-output_file", help="output of value pickle file", required=True)
	args = parser.parse_args()

	with open(args.loading_file, "r") as f:
		loading_configs =  json.load(f)

	with open(args.model_file, "r") as f:
		model_config =  json.load(f)


	dataloader = make_dataloader(loading_configs)

	model = make_model(model_config)

	ensemble_values_corpus = calculate_values(model, dataloader)

	save_corpus(ensemble_values_corpus, args.output_file)

	


if __name__ == '__main__':
	main()
