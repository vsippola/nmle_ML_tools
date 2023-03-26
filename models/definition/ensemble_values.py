# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-01-31      #
# --------------------------------- #

import torch
from .custom_nn_module import CustomNNModule

class EnsembleValuesBlock(CustomNNModule):
	
	def __init__(self, *args, **kwargs):				

		self.input_dimension = 1
		self.ensemble_batch_key = kwargs.pop("ensemble_batch_key")
		self.output_key = kwargs.pop("output_key")
		self.logits_keys = kwargs.pop("logits_keys")
		
		#call to custom NN module
		super(EnsembleValuesBlock, self).__init__(*args, **kwargs)

		#this is not a module list. These parameters will not be exposed directly to an optimizer called on the GranularModel object
		self.ensemble_models = {}	
		self.output_dimension = None

	
	def set_ensemble_models(self, ensemble_models):
		self.ensemble_models = ensemble_models
		

	def forward(self, state_object):

		ensemble_batch = state_object.pop(self.ensemble_batch_key)

		logits_list = []

		for model_key in ensemble_batch:

			model = self.ensemble_models[model_key]
			model_batch = ensemble_batch[model_key]
			model_logits = model(model_batch)[self.logits_keys[model_key]]
			logits_list.append(model_logits)

		ensemble_logits = torch.stack(logits_list, dim=1)

		ensemble_values = torch.nn.functional.softmax(ensemble_logits, dim=2)

		state_object[self.output_key] = ensemble_values

		return state_object


	def to_device(self):
		
		for model_key, model in self.ensemble_models.items():
			model.to_device()


	def get_param_dict(self):

		param_dict={
			"state_dict":""
		}

		return param_dict


	def load_from_dict(self, state_dict):	
		
		pass

		


