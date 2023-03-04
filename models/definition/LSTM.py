# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2021-06-16      #
# --------------------------------- #

import torch

from .custom_nn_module import CustomNNModule


class LSTMBlock(CustomNNModule):
	
	def __init__(self, *args, **kwargs):		

		self.input_dimension = kwargs.pop("input_dimension")

		hidden_dimension = kwargs.pop("hidden_dimension", self.input_dimension)
		num_layers = kwargs.pop("num_layers")
		bias = kwargs.pop("bias", True)
		self.batch_first = kwargs.pop("batch_first", True)
		dropout = kwargs.pop("dropout", 0.0)
		bidirectional = kwargs.pop("bidirectional", True)
		proj_size = kwargs.pop("proj_size", 0)

		self.padding_value = kwargs.pop("padding_value", 0.0)	
		
		self.padded_sentences_key = kwargs.pop("padded_sentences_key")
		self.output_key = kwargs.pop("output_key")		
		self.sentence_length_key = kwargs.pop("sentence_length_key")	

		super(LSTMBlock, self).__init__(*args, **kwargs)

		self.output_dimension = (2 if bidirectional else 1)*hidden_dimension

		self.LSTM = torch.nn.LSTM(
			input_size = self.input_dimension, 
			hidden_size = hidden_dimension, 
			num_layers = num_layers,
			bias = bias,
			batch_first = self.batch_first,
			dropout = dropout, 
            bidirectional = bidirectional, 
            proj_size = proj_size
            )

		
	

	#input/output is expected as any linear layer N, *, input_dimension -> N, *, output_dimension
	def forward(self, state_object):

		padded_sentences_list = state_object[self.padded_sentences_key]
		padded_sentences_list = padded_sentences_list.to(self.device)

		sentence_lengths = state_object[self.sentence_length_key]

		padded_sentences_list = torch.nn.utils.rnn.pack_padded_sequence(padded_sentences_list, sentence_lengths, batch_first=self.batch_first, enforce_sorted=False)	

		padded_sentences_list = self.LSTM(padded_sentences_list)[0]  #packed output # seqlen x batch x 2*nhid	

		#similarly this function now unsorts the sentence for us
		padded_sentences_list = torch.nn.utils.rnn.pad_packed_sequence(padded_sentences_list, batch_first=self.batch_first, padding_value=self.padding_value, total_length=None)[0]		

		state_object[self.output_key] = padded_sentences_list

		return state_object