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
import torch.nn as nn
from .custom_nn_module import CustomNNModule


class Word2VecBlock(CustomNNModule):
	
	"""
	kwargs
	vecs
	poolingtype
	device
	"""
	def __init__(self, *args, **kwargs):

		asset("poolingtype" in kwargs)

		vecs = kwargs.pop("vecs")
		self.poolingtype = kwargs.pop("poolingtype")

		super(Word2VecBlock, self).__init__(*args, **kwargs)

		self.input_size = len(vecs[0])
		self.output_size = self.input_size

		with torch.no_grad():

			self.freeze = True
	
			self.embedding = nn.EmbeddingBag.from_pretrained(vecs, mode=self.poolingtype, freeze=self.freeze, sparse=True, device=self.device)


	def forward(self, indexs, offsets=None):

		indexs = indexs.to(self.device)
		offests = offsets.to(self.device)
		
		return self.embedding(indexs, offests)



	def get_param_dict(self):
		param_dict={
			"state_dict":"" if self.freeze else self.state_dict()
		}
		return param_dict


	def load_from_dict(self, state_dict):	
		if not self.freeze:
			self.load_state_dict(state_dict)