# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

import sys

import torch

from .context import definition
from definition.ensemble import EnsembleBlock



class EnsembleBuilder():
	
	def __init__(self, *args, **kwargs):

		self.configured = False


	def configure(self, *args, **kwargs):

		number_classes = kwargs.pop("number_classes")
		number_models = kwargs.pop("number_models")
		non_det_models = kwargs.pop("non_det_models", False)
		noise_std = kwargs.pop("noise_std", None)

		self.configured = True
		
		self.non_det_models = non_det_models
		self.noise_std = noise_std
		self.number_classes = number_classes
		self.number_models = number_models
		self.ensemble_params = kwargs


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		
		self.ensemble_params["input_dimension"] = self.number_classes
		
		ensemble_block = EnsembleBlock(**self.ensemble_params)

		ensemble_weights = [torch.full((self.number_models,), fill_value=0.5)]

		if self.non_det_models:
			ensemble_weights += [torch.full((self.number_models,), fill_value=0.5) for _ in range(self.number_models)]

			mul_mask = [torch.tensor([1 if j!=i else 0 for j in range(self.number_models)]) for i in range(self.number_models + 1)]
			add_mask = [torch.tensor([0 if j!=i else -1e9 for j in range(self.number_models)]) for i in range(self.number_models + 1)]

			ensemble_block.set_mask((mul_mask, add_mask))

		if self.noise_std is not None:

			noise_generator_param = [torch.tensor([0.0]),torch.tensor([self.noise_std])]

			ensemble_block.set_noise_generator_param(noise_generator_param)

		ensemble_block.set_ensemble_weights(ensemble_weights)

		return ensemble_block

					


