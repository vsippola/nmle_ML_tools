# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

"""
This class builds a word2vec NN object.10

It loads vectors from a given pkl file.

"""

from .dataset_objects import SentencePairDataset

import os
import pickle
import sys

class SentencePairDatasetBuilder():
	
	def __init__(self, *args, **kwargs):
		
		self.configured = False


	def configure(self, *args, **kwargs):

		#update configuration
		corpus_pkl_file = kwargs.pop("corpus_pkl_file")
		
		if not (os.path.isfile(corpus_pkl_file)):
			print()
			print(f'file {corpus_pkl_file} does not exist')
			sys.exit()

		self.configured = True
		
		self.corpus_pkl_file = corpus_pkl_file
		self.dataloader_params = kwargs			


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		dataset = SentencePairDataset(**self.dataloader_params)

		with open(self.corpus_pkl_file, 'rb') as f:
			corpus = pickle.load(f)

		dataset.set_corpus(corpus)
		
		return dataset







		

					


