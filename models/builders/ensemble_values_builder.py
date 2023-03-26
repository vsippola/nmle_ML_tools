# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

import sys

import torch

from .context import definition
from definition.ensemble_values import EnsembleValuesBlock



class EnsembleValuesBuilder():
	
	def __init__(self, *args, **kwargs):

		self.configured = False


	def configure(self, *args, **kwargs):

		ensemble_model_configs = kwargs.pop("ensemble_model_configs")

		self.configured = True
		
		self.ensemble_model_configs = ensemble_model_configs
		self.ev_params = kwargs
			
	"""
	This function should probably do a forward pass on the input/output keys and throw an error if there is ever a key missing
	However this requires changing the way input/outut keys are used

	TODO do this
	"""
	def build(self):

		#this is here to avoid a circular dependancy?
		from models.builders.module_factory import ModuleFactory

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		
		ensemble_models = torch.nn.ModuleDict()

		for model_key, model_config in self.ensemble_model_configs.items():

			model = ModuleFactory.BUILD_MODULE(**model_config)
			ensemble_models[model_key] = model
			output_dimension = model.output_dimension

		ensemble_values_block = EnsembleValuesBlock(**self.ev_params)
		ensemble_values_block.output_dimension = output_dimension

		ensemble_values_block.set_ensemble_models(ensemble_models)

		return ensemble_values_block

					


