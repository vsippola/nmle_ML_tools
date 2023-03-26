# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

import json
import sys

import torch

from .context import definition


class PretrainedModelBuilder():
	
	def __init__(self, *args, **kwargs):
		
		self.configured = False


	def configure(self, *args, **kwargs):

		model_config_file = kwargs.pop("model_config_file")
		checkpoint_file = kwargs.pop("checkpoint_file", None)

		self.configured = True

		self.model_config_file = model_config_file
		self.checkpoint_file = checkpoint_file


	def build(self):

		from models.builders.module_factory import ModuleFactory

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		with open(self.model_config_file, "r") as f:
			model_config = json.load(f)

		model = ModuleFactory.BUILD_MODULE(**model_config)

		if self.checkpoint_file is not None:
			checkpoint = torch.load(self.checkpoint_file)
			model.load_from_dict(**checkpoint["model_params"])

		return model






		

					


