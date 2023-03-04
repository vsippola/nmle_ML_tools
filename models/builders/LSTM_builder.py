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
from definition.LSTM import LSTMBlock

import sys

class LSTMBuilder():
	
	def __init__(self, *args, **kwargs):
		
		self.configured = False


	"""
	kwargs
	vector_pkl_file - file where the matrix of vectors is stored
	word2vec_config - parameter dictionary provided for the word2vec NN object from configuration
	"""
	def configure(self, *args, **kwargs):


		self.configured = True
		
		self.LSTM_params = kwargs		
			


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		LSTM_block = LSTMBlock(**self.LSTM_params)

		return LSTM_block






		

					


