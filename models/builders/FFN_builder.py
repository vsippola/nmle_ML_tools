# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

from .context import definition
from definition.FFN import FFNBlock

import sys

import torch
import torch.nn as nn

class FFNBuilder():
	
	def __init__(self, *args, **kwargs):
		self.configured = False


	"""
	kwargs
	vector_pkl_file - file where the matrix of vectors is stored
	word2vec_config - parameter dictionary provided for the word2vec NN object from configuration
	"""
	def configure(self, *args, **kwargs):

		#update configuration
		dropout_probability = kwargs.pop("dropout_probability", 0.0)
		output_dimensions = kwargs.pop("output_dimensions")
		bias = kwargs.pop("bias", True)
		drop_last = kwargs.pop("drop_last", True)

		if len(output_dimensions) == 0:
			print()
			print("no output dimensions specified")
			sys.exit()

		self.configured = True
		self.FFN_params = kwargs	
		self.dropout_probability = dropout_probability	
		self.output_dimensions = output_dimensions
		self.bias = bias
		self.drop_last = drop_last


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		self.FFN_params["output_dimension"] = self.output_dimensions[-1]

		ffn_block = FFNBlock(**self.FFN_params)

		dimension_list = [ffn_block.input_dimension] + self.output_dimensions
		modules = []

		for d_i in range(len(dimension_list) - 1):
			d1 = dimension_list[d_i]
			d2 = dimension_list[d_i + 1]

			modules.append(nn.Dropout(self.dropout_probability))
			modules.append(nn.Linear(d1, d2, bias=self.bias))
			modules.append(nn.GELU())
			#modules.append(nn.ReLU())

		if self.drop_last:
			modules = modules[:-1] # get rid of trailing activation function

		ffn_block.set_sequential(nn.Sequential(*modules))

		return ffn_block







		

					


