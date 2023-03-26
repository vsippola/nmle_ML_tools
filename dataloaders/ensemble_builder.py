# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-03-19      #
# --------------------------------- #

import os
import pathlib
import pickle
import sys

from .collate_fns import CollateFunctions
from .ensemble import EnsembleDataset
from .label_transform_fns import LabelTransformFunctions

class EnsembleDatasetBuilder():

	
	def __init__(self, *args, **kwargs):
		
		self.configured = False


	def configure(self, *args, **kwargs):

		#update configuration
		corpus_file = kwargs.pop("corpus_file", None)
		corpus = kwargs.pop("corpus", None)

		dataset_configs = kwargs.pop("dataset_configs")

		example_number_index = kwargs.pop("example_number_index", None)
		label_transform_params = kwargs.pop("label_transform_params", None)
		collate_fn_params = kwargs.pop("collate_fn_params", None)

		if (corpus_file is None) and (corpus is None):
			print()
			print("Either a corpus file or a list of corpus text example lines must be provided")
			sys.exit()
		
		if (corpus_file is not None) and (not (os.path.isfile(corpus_file))):
			print()
			print(f'file {corpus_file} does not exist')
			sys.exit()		

		self.configured = True

		self.corpus_file = corpus_file
		self.corpus = corpus
		self.dataset_configs = dataset_configs
		self.example_number_index = example_number_index
		self.label_transform_params = label_transform_params
		self.collate_fn_params = collate_fn_params
		


	def build(self):

		from .dataset_factory import DatasetFactory

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		#load corpus if required
		if self.corpus_file is None:

			corpus = self.corpus

		else:
			
			corpus = []
			
			with open(self.corpus_file, 'r') as f:

				for example_line in f:
					corpus.append(example_line)

		#get the index tranformation function
		if self.example_number_index is not None:

			def example_number_tranform_fn(example):

				return int(example[self.example_number_index])

		else:

			example_number_tranform_fn = None


		#get the label transformation function
		if self.label_transform_params is not None:

			label_transform_fn = LabelTransformFunctions.get_fn(**self.label_transform_params)		

		else:

			label_transform_fn = None


		#define the tranform function
		def transform_fn(example):

			#should this be a chosen fns too? probably
			example = example.strip("\n").split("\t")

			example_number = example_number_tranform_fn(example) if example_number_tranform_fn is not None else -1
			label = label_transform_fn(example) if label_transform_fn is not None else -1			

			return [example_number, label]


		datasets = {}
		collate_fns = {}

		for dataset_config_key, dataset_config  in self.dataset_configs.items():

			dataset_config["corpus"] = corpus
			dataset = DatasetFactory.BUILD_DATASET(**dataset_config)
			datasets[dataset_config_key] = dataset
			collate_fns[dataset_config_key] = dataset.collate_fn


		self.collate_fn_params["collate_fns"] = collate_fns

		collate_fn = CollateFunctions.get_fn(**self.collate_fn_params)

		ensemble_dataset = EnsembleDataset()

		ensemble_dataset.set_corpus(corpus)
		ensemble_dataset.set_collate_fn(collate_fn)
		ensemble_dataset.set_datasets(datasets)
		ensemble_dataset.set_transform_fn(transform_fn)
	

		return ensemble_dataset
					
