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

		number_classes = kwargs.get("number_classes")
		number_models = kwargs.get("number_models")

		self.configured = True
		
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

		ensemble_weights = torch.full((1,self.number_models), fill_value=0.5)
		ensemble_block.set_ensemble_weights(ensemble_weights)

		return ensemble_block

					


