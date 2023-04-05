# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-01-31      #
# --------------------------------- #

import torch
from .custom_nn_module import CustomNNModule

class EnsembleMLPBlock(CustomNNModule):
	
	def __init__(self, *args, **kwargs):				

		self.input_dimension = kwargs.pop("input_dimension")
		self.ensemble_values_key = kwargs.pop("ensemble_values_key")
		self.output_key = kwargs.pop("output_key")
		self.class_percentages_key = kwargs.pop("class_percentages_key")
		self.mlp_weights_key = kwargs.pop("mlp_weights_key")
		
		super(EnsembleMLPBlock, self).__init__(*args, **kwargs)

		self.MLP = None

		self.output_dimension = self.input_dimension

	
	def set_MLP(self, MLP):			
		self.MLP = MLP


	def forward(self, state_object):

		ensemble_values = state_object[self.ensemble_values_key]
		ensemble_values = ensemble_values.to(self.device)

		flat_ensemble_values = torch.flatten(ensemble_values, start_dim=1)

		weights = self.MLP({self.mlp_weights_key:flat_ensemble_values})[self.mlp_weights_key]

		weights = torch.nn.functional.softmax(weights, dim=1).unsqueeze(2)

		weighted_values = torch.mul(ensemble_values, weights)

		output_logits = torch.sum(weighted_values, dim=1)

		state_object[self.output_key] = output_logits
		state_object[self.class_percentages_key] = output_logits

		return state_object
