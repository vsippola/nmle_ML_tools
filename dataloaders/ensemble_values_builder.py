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
from .indexer import IndexerDataset


class EnsembleValuesDatasetBuilder():

	
	def __init__(self, *args, **kwargs):
		
		self.configured = False


	def configure(self, *args, **kwargs):

		#update configuration
		corpus_file = kwargs.pop("corpus_file", None)
		corpus = kwargs.pop("corpus", None)

		example_number_index = kwargs.pop("example_number_index", 0)
		label_index = kwargs.pop("label_index", 1)
		values_index = kwargs.pop("values_index", 2)
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
		self.index_transform = [example_number_index, label_index, values_index]
		self.collate_fn_params = collate_fn_params
		


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		#load corpus if required
		if self.corpus_file is None:

			corpus = self.corpus

		else:
			
			with open(self.corpus_file, 'rb') as f:

				corpus = pickle.load(f)


		def transform_fn(example):

			return [example[i] for i in self.index_transform]


		if self.collate_fn_params is not None:

			collate_fn = CollateFunctions.get_fn(**self.collate_fn_params)

		else:

			collate_fn = None

		dataset = IndexerDataset()

		dataset.set_corpus(corpus)		
		dataset.set_transform_fn(transform_fn)		
		dataset.set_collate_fn(collate_fn)

		return dataset
					
