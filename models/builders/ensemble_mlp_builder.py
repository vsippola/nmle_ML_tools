# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

import sys

import torch

from .context import definition
from definition.ensemble_mlp import EnsembleMLPBlock



class EnsembleMLPBuilder():
	
	def __init__(self, *args, **kwargs):

		self.configured = False


	def configure(self, *args, **kwargs):

		number_classes = kwargs.pop("number_classes")
		number_models = kwargs.pop("number_models")
		mlp_config = kwargs.pop("mlp_config")
		mlp_weights_key = kwargs.get("mlp_weights_key")

		self.configured = True

		self.mlp_weights_key = mlp_weights_key
		self.mlp_config = mlp_config
		self.number_classes = number_classes
		self.number_models = number_models
		self.ensemble_params = kwargs


	def build(self):

		from models.builders.module_factory import ModuleFactory

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()		

		
		self.ensemble_params["input_dimension"] = self.number_classes
		
		ensemble_mlp_block = EnsembleMLPBlock(**self.ensemble_params)

		self.mlp_config["input_dimension"] = self.number_classes*self.number_models
		self.mlp_config["vector_batch_key"] = self.mlp_weights_key
		self.mlp_config["output_key"] = self.mlp_weights_key
		MLP = ModuleFactory.BUILD_MODULE(**self.mlp_config)

		ensemble_mlp_block.set_MLP(MLP)

		return ensemble_mlp_block

					


