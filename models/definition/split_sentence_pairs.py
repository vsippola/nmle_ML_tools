# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10       #
#       Created on: 2023-02-13      #
# --------------------------------- #

from .custom_nn_module import CustomNNModule


class SplitSentencePairs(CustomNNModule):


	def __init__(self, *args, **kwargs):

		self.input_dimension = kwargs.pop("input_dimension")
		self.output_dimension = self.input_dimension

		self.flat_tensor_key = kwargs.pop("flat_tensor_key")
		self.output1_key = kwargs.pop("output1_key")	
		self.output2_key = kwargs.pop("output2_key")			

		super(SplitSentencePairs, self).__init__(*args, **kwargs)
		

	def forward(self, state_object):

		flat_sentence_tensor = state_object.pop(self.flat_tensor_key)
		
		state_object[self.output1_key] = flat_sentence_tensor[:len(flat_sentence_tensor)//2]
		state_object[self.output2_key] = flat_sentence_tensor[len(flat_sentence_tensor)//2:]

		return state_object




	
