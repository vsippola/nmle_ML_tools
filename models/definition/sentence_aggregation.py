# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10       #
#       Created on: 2023-02-13      #
# --------------------------------- #

import torch
from .custom_nn_module import CustomNNModule


class SentenceAggregationBlock(CustomNNModule):

	def _MaxPoolingSentenceAggregationStrategy(self, sentences, sentence_lengths):
		
		sentences, dimensions = torch.max(sentences, 1)

		return sentences


	def _MeanPoolingSentenceAggregationStrategy(self, sentences, sentence_lengths):
		
		sentences = torch.sum(sentences, 1)
		sentence_lengths = torch.tensor(sentence_lengths, device=torch.device(self.device))
		sentence_lengths = sentence_lengths.unsqueeze(1)

		sentences = sentences / sentence_lengths

		return sentences


	def _SumPoolingSentenceAggregationStrategy(self, sentences, sentence_lengths):
		
		sentences = torch.sum(sentences, 1)

		return sentences, []


	STRATEGY = {"maxpooling":_MaxPoolingSentenceAggregationStrategy, \
				"meanpooling":_MeanPoolingSentenceAggregationStrategy, \
				"sumpooling":_SumPoolingSentenceAggregationStrategy \
	}


	def __init__(self, *args, **kwargs):

		self.input_dimension = kwargs.pop("input_dimension")
		self.output_dimension = self.input_dimension

		self.padded_sentences_key = kwargs.pop("padded_sentences_key")
		self.output_key = kwargs.pop("output_key")		
		self.sentence_length_key = kwargs.pop("sentence_length_key")

		super(SentenceAggregationBlock, self).__init__(*args, **kwargs)


	def set_sentence_aggregation_strategy(self, strategy):

		self.sentence_aggregation_strategy = strategy


	def forward(self, state_object):

		padded_sentences_list = state_object[self.padded_sentences_key]
		padded_sentences_list = padded_sentences_list.to(self.device)

		sentence_lengths = state_object[self.sentence_length_key]

		aggregated_sentences = self.sentence_aggregation_strategy(self, padded_sentences_list, sentence_lengths)

		state_object[self.output_key] = aggregated_sentences

		return state_object




	
