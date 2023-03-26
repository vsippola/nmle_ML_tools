# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-01-31      #
# --------------------------------- #

import torch
from .custom_nn_module import CustomNNModule

class EnsembleBlock(CustomNNModule):
	
	def __init__(self, *args, **kwargs):				

		self.input_dimension = kwargs.pop("input_dimension")
		self.ensemble_values_key = kwargs.pop("ensemble_values_key")
		self.output_key = kwargs.pop("output_key")

		#overwrite defaults
		kwargs["freeze"] = kwargs.pop("freeze", True)
		
		super(EnsembleBlock, self).__init__(*args, **kwargs)

		self.ensemble_weights = None
		self.output_dimension = self.input_dimension

	
	def set_ensemble_weights(self, ensemble_weights):
		self.ensemble_weights = torch.nn.Parameter(ensemble_weights, requires_grad=not self.freeze)
		

	def forward(self, state_object):

		ensemble_values = state_object[self.ensemble_values_key]
		ensemble_values = ensemble_values.to(self.device)

		softmax_ensemble_weights = torch.nn.functional.softmax(self.ensemble_weights, dim=1)

		weighted_values = torch.mul(ensemble_values, softmax_ensemble_weights)

		output_logits = torch.sum(weighted_values, dim=1)

		state_object[self.output_key] = output_logits

		return state_object

