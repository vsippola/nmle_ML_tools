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
from definition.sentence_aggregation import SentenceAggregationBlock

import sys

class SentenceAggregationBuilder():
	
	def __init__(self, *args, **kwargs):
		
		self.configured = False


	"""
	kwargs
	vector_pkl_file - file where the matrix of vectors is stored
	word2vec_config - parameter dictionary provided for the word2vec NN object from configuration
	"""
	def configure(self, *args, **kwargs):

		#update configuration
		agregation_type = kwargs.pop("aggregation_type")

		if agregation_type not in SentenceAggregationBlock.STRATEGY:
			print(f"Aggregation type not found {agregation_type}")
			sts.exit()

		self.configured = True
		
		self.SA_params = kwargs
		self.aggregation_function = SentenceAggregationBlock.STRATEGY[agregation_type]
			


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		sentence_aggregation_block = SentenceAggregationBlock(**self.SA_params)
		sentence_aggregation_block.set_sentence_aggregation_strategy(self.aggregation_function)

		return sentence_aggregation_block






		

					


