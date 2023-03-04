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

		if device is None:
			
			device = 'cuda' if torch.cuda.is_available() else 'cpu'

		self.device = device

	def to_device(self):
		self.to(self.device)

	def get_param_dict(self):
		param_dict={
			"state_dict":self.state_dict()
		}
		return param_dict

	def load_from_dict(self, state_dict):	
		self.load_state_dict(state_dict)