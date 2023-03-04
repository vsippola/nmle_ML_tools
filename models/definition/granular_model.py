# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-01-31      #
# --------------------------------- #

"""
Top level model.

Stores a list of N blocks, i.e. your chosen granularity

During training/evaluation the list is wrapped in a sequentual. Your outputs/inputs must be compatible between each block.
 
During probing or XAI you can specify 0 <= ki <= N to retrieve te outputs for the kith level of decision making

You can also specify (input, k) to get the output using input starting from the kth step

"""
import torch
from .custom_nn_module import CustomNNModule

#parameters
#input_dimension - dimension of input
#output_dimension - dimension of output
#device - cpu or cuda

"""
kwargs
input_dimension
output_dimension
device
"""
class GranularModel(CustomNNModule):
	
	def __init__(self, *args, **kwargs):				

		self.input_dimension = kwargs.pop("input_dimension")
		
		#call to custom NN module
		super(GranularModel, self).__init__(*args, **kwargs)

		#this is not a module list. These parameters will not be exposed directly to an optimizer called on the GranularModel object
		self.nn_block_list = []		
		self.output_dimension = None

	
	#to be set if you only want end to end behaviour such as during training or evaluation on the testing set
	def set_sequential(self):

		self.sequential = torch.nn.Sequential(*self.nn_block_list)
		self.forward = self.sequential.forward


	def to_device(self):
		
		for nn_block in self.nn_block_list:
			nn_block.to_device()


	def get_param_dict(self):
		
		state_dict_list = []

		for nn_block in self.nn_block_list:
			state_dict_list.append(nn_block.get_param_dict())

		param_dict={
			"state_dict_list":state_dict_list
		}

		return param_dict


	def load_from_dict(self, state_dict_list):	
		
		for sd_i, state_dict in enumerate(state_dict_list):
			self.nn_block_list[sd_i].load_from_dict(**state_dict)

	
	#TODO, granular outputs, start from (input, K) for use in probings/explaining


