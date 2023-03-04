# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2021-06-16      #
# --------------------------------- #

"""
for simplicity this only does a FFN operation with ReLU and (optionally) dorpout between layerer
the last layer of logits is untouched. This allows the module to be used

1) if you want to softmax
2) if you want to do regression
3) if two different  outputs are used together such as for cosine similarity

Your builder should confifure the FFN to your specifications


"""

from .custom_nn_module import CustomNNModule

#parameters
#input_dimension - dimension of input
#output_dimension - dimension of output
#dropout_prob - the probability a dimension will be 0'd (ie at 0.0 keep all dimension)
#hidden_dimensions - should be empty if isLiner is True, otherwise contains the values for the sizes of hidden layers
#device - cpu or cuda

"""
kwargs
input_dimension
output_dimension
dropout_probability=0.0
hidden_dimensions=[]
isLinear=True
device
"""
class FFNBlock(CustomNNModule):
	
	def __init__(self, *args, **kwargs):		

		self.input_dimension = kwargs.pop("input_dimension")
		self.output_dimension = kwargs.pop("output_dimension")

		self.vector_batch_key = kwargs.pop("vector_batch_key")
		self.output_key = kwargs.pop("output_key")

		super(FFNBlock, self).__init__(*args, **kwargs)		
		
		

	def set_sequential(self, sequential):

		self.FFN = sequential
	

	#input/output is expected as any linear layer N, *, input_dimension -> N, *, output_dimension
	def forward(self, state_object):

		vector_batch = state_object.pop(self.vector_batch_key)
		vector_batch = vector_batch.to(self.device)

		state_object[self.output_key] = self.FFN(vector_batch)

		return state_object