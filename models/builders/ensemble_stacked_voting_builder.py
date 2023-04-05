# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

import sys

import torch

from .context import definition
from definition.ensemble_stacked_voting import EnsembleStackedVotingBlock



class EnsembleStackedVotingBuilder():
	
	def __init__(self, *args, **kwargs):

		self.configured = False


	def configure(self, *args, **kwargs):

		number_classes = kwargs.pop("number_classes")

		self.configured = True

		self.number_classes = number_classes
		self.ensemble_params = kwargs


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		
		self.ensemble_params["input_dimension"] = self.number_classes
		
		ensemble_block = EnsembleStackedVotingBlock(**self.ensemble_params)		

		return ensemble_block
			


