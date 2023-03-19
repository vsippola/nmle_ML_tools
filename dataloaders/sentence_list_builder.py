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
from .label_transform_fns import LabelTransformFunctions
from .sentence_list import SentenceListDataset
from .sentence_transform_fns import SentenceTransformFunctions

class SentenceListDatasetBuilder():

	
	def __init__(self, *args, **kwargs):
		
		self.configured = False


	def configure(self, *args, **kwargs):

		#update configuration
		corpus_file = kwargs.pop("corpus_file", None)
		corpus = kwargs.pop("corpus", None)

		example_number_index = kwargs.pop("example_number_index", None)
		label_transform_params = kwargs.pop("label_transform_params", None)
		sentence_transform_params = kwargs.pop("sentence_transform_params", None)
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
		self.example_number_index = example_number_index
		self.label_transform_params = label_transform_params
		self.sentence_transform_params = sentence_transform_params
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


		#get the label transformation function
		if self.sentence_transform_params is not None:

			sentence_transform_fn = SentenceTransformFunctions.get_fn(**self.sentence_transform_params)		

		else:

			sentence_transform_fn = None


		#define the tranform function
		def transform_fn(example):

			#should this be a chosen fns too? probably
			example = example.strip("\n").split("\t")

			example_number = example_number_tranform_fn(example) if example_number_tranform_fn is not None else None
			label = label_transform_fn(example) if label_transform_fn is not None else None
			sentences = sentence_transform_fn(example) if sentence_transform_fn is not None else None

			return [example_number, label, sentences]


		if self.collate_fn_params is not None:

			collate_fn = CollateFunctions.get_fn(**self.collate_fn_params)

		else:

			collate_fn = None

		dataset = SentenceListDataset()

		dataset.set_corpus(corpus)
		dataset.set_transform_fn(transform_fn)
		dataset.set_collate_fn(collate_fn)

		return dataset
					
