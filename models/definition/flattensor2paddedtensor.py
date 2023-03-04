# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10       #
#       Created on: 2023-02-13      #
# --------------------------------- #

import torch
from .custom_nn_module import CustomNNModule


class FlatTensor2PaddedTensor(CustomNNModule):


	def __init__(self, *args, **kwargs):

		self.input_dimension = kwargs.pop("input_dimension")
		self.output_dimension = self.input_dimension

		self.padding_value = kwargs.pop("padding_value")

		self.flat_tensor_key = kwargs.pop("flat_tensor_key")
		self.output_key = kwargs.pop("output_key")	
		self.sentence_length_key = kwargs.pop("sentence_length_key")			

		super(FlatTensor2PaddedTensor, self).__init__(*args, **kwargs)
		

	def forward(self, state_object):

		flat_sentence_tensor = state_object.pop(self.flat_tensor_key)
		sentence_lengths = state_object[self.sentence_length_key]

		sentence_sequences = torch.split(flat_sentence_tensor, sentence_lengths, dim=0)

		padded_sentence_tensor = torch.nn.utils.rnn.pad_sequence(sentence_sequences, batch_first=True, padding_value=self.padding_value)

		state_object[self.output_key] = padded_sentence_tensor

		return state_object




	
