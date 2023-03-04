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

from .context import definition
from definition.word2vec import Word2VecBlock

import os
import pickle
import sys

class Word2VecBuilder():
	
	def __init__(self, *args, **kwargs):
		
		self.configured = False


	def configure(self, *args, **kwargs):

		#update configuration
		vector_pkl_file = kwargs.pop("vector_pkl_file")
		
		if not (os.path.isfile(vector_pkl_file)):
			print()
			print(f'file {vector_pkl_file} does not exist')
			sys.exit()

		self.configured = True
		
		self.vector_pkl_file = vector_pkl_file
		self.word2vec_params = kwargs			


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		w2v_block = Word2VecBlock(**self.word2vec_params)

		with open(self.vector_pkl_file, 'rb') as f:
			vecs = pickle.load(f)

		w2v_block.set_vectors(vecs)

		return w2v_block







		

					


