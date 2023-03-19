# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-28      #
# --------------------------------- #

"""
This class provides a modules level wrapper for our word dictionaries

For small sized dictionaries it allows us to store them in GPU

Can allow for training some parameters

It is assumes the corpus will be turned into indexes for use with this object

params
vecs
device gpu, cpu or None
"""

import torch
from .custom_nn_module import CustomNNModule


class Word2VecBlock(CustomNNModule):
	
	"""
	kwargs
	input_key - key to get input from the state object
	output_key - key to save the output from in the state object
	freeze (optional) True or False
	device
	"""
	def __init__(self, *args, **kwargs):

		self.index_key = kwargs.pop("index_key")
		self.output_key = kwargs.pop("output_key")
		#override default
		self._freeze = kwargs.pop("freeze", True)	
		kwargs["freeze"] = self._freeze

		super(Word2VecBlock, self).__init__(*args, **kwargs)

		self.input_dimension = 1		


	def set_vectors(self, vecs):

		self.output_dimension = len(vecs[0])

		with torch.no_grad():	

			self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor(vecs), freeze=self._freeze, sparse=True)	



	def forward(self, state_object):

		indexes = state_object.pop(self.index_key)

		indexes = indexes.to(self.device)

		state_object[self.output_key] = self.embedding(indexes)
		
		return state_object



	def get_param_dict(self):
		param_dict={
			"state_dict":"" if self.freeze else self.state_dict()
		}
		return param_dict


	def load_from_dict(self, state_dict):	
		if not self.freeze:
			self.load_state_dict(state_dict)