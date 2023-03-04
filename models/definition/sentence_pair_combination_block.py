# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10       #
#       Created on: 2023-01-16      #
# --------------------------------- #

import torch
from .custom_nn_module import CustomNNModule


class SentencePairCombinationBlock(CustomNNModule):

	def _concatenation(self, s1, s2):
		return [s1, s2]

	def _absolute_difference(self, s1, s2):
		return [torch.abs(s1 - s2)]

	def _pairwise_multiplication(self, s1, s2):
		return [s1 * s2]

	COMBINATION_TYPES = {
		"concatenation":_concatenation,
		"absolute_difference":_absolute_difference,
		"pairwise_multiplication":_pairwise_multiplication
	}

	INPUT_MULTIPLIER = {
		"concatenation":2,
		"absolute_difference":1,
		"pairwise_multiplication":1
	}


	def __init__(self, *args, **kwargs):

		self.input_dimension = kwargs.pop("input_dimension")
		
		self.vector_batch1_key = kwargs.pop("vector_batch1_key")
		self.vector_batch2_key = kwargs.pop("vector_batch2_key")
		self.output_key = kwargs.pop("output_key")

		super(SentencePairCombinationBlock, self).__init__(*args, **kwargs)

		self.output_dimension = None


	def set_combinations(self, combination_function_list):

		self.combination_functions = combination_function_list


	def forward(self, state_object):

		s1 = state_object.pop(self.vector_batch1_key)
		s2 = state_object.pop(self.vector_batch2_key)

		combined_sentences = []

		for combination_function in self.combination_functions:
			combined_sentences += combination_function(self, s1, s2)

		combined_sentences = torch.cat(combined_sentences, 1)

		state_object[self.output_key] = combined_sentences

		return state_object
