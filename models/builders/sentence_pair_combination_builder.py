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
from definition.sentence_pair_combination_block import SentencePairCombinationBlock

import sys

class SentencePairCombinationBuilder():
	
	def __init__(self, *args, **kwargs):

		self.configured = False


	"""
	kwargs
	vector_pkl_file - file where the matrix of vectors is stored
	word2vec_config - parameter dictionary provided for the word2vec NN object from configuration
	"""
	def configure(self, *args, **kwargs):

		combination_type_list = kwargs.pop("combination_type_list")

		if len(combination_type_list) == 0:
			print()
			print("no sentence combaintion types provided")
			sys.exit()

		self.configured = True
		self.SPC_params = kwargs
		self.combination_type_list = combination_type_list
			

	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()		

		SPC_block = SentencePairCombinationBlock(**self.SPC_params)

		input_multiplier = 0
		SPC_function_list = []

		for combination_type in self.combination_type_list:
			input_multiplier += SentencePairCombinationBlock.INPUT_MULTIPLIER[combination_type]
			SPC_function_list.append(SentencePairCombinationBlock.COMBINATION_TYPES[combination_type])

		SPC_block.output_dimension = SPC_block.input_dimension * input_multiplier
		SPC_block.set_combinations(SPC_function_list)

		return SPC_block






		

					


