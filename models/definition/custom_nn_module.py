# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-1-28      #
# --------------------------------- #

"""
Custom module for functionality we want in all the pieces of our model
"""

import torch
import torch.nn as nn

class CustomNNModule(nn.Module):
	
	def __init__(self, *args, **kwargs):
		super(CustomNNModule, self).__init__()

		device = kwargs.pop("device", None)
		freeze = kwargs.pop("freeze", False)

		if device is None:
			
			device = 'cuda' if torch.cuda.is_available() else 'cpu'

		self.device = device
		self._freeze = freeze

	def to_device(self, device=None):
		if device is not None:
			self.device = device

		self.to(self.device)

	def get_param_dict(self):
		param_dict={
			"state_dict":self.state_dict()
		}
		return param_dict

	def load_from_dict(self, state_dict):	
		self.load_state_dict(state_dict)

	def freeze(self, freeze=None):
		if freeze is not None:
			self._freeze = freeze

		for param in model.parameters():
			param.requires_grad = not self._freeze
