# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-03-10      #
# --------------------------------- #

import torch

from .custom_nn_module import CustomNNModule


class HFModel(CustomNNModule):
	
	def __init__(self, *args, **kwargs):	

		self.batch_key = kwargs.pop("batch_key")	
		self.output_key = kwargs.pop("output_key")	
		self.output_dimension  = kwargs.pop("output_dimension")

		self.logits_transform = kwargs.pop("logits_transform", None)
		if self.logits_transform is not None:
			self.logits_transform  = torch.tensor(self.logits_transform )

		self.input_dimension = 1
		
		super(HFModel, self).__init__(*args, **kwargs)	
		
		self.output_dimension = None

		self.model = None


	def set_model(self, model):		

		self.model = model


	def set_get_logits_fn(self, get_logits_fn):

		self.get_logits_fn = get_logits_fn


	def to_device(self):

		self.to(self.device)

		if self.logits_transform is not None:
			self.logits_transform = self.logits_transform.to(self.device)
			

	def forward(self, state_object):

		bert_batch = state_object.pop(self.batch_key)
		bert_batch['input_ids'] = bert_batch['input_ids'].to(self.device)
		bert_batch['attention_mask'] = bert_batch['attention_mask'].to(self.device)

		output_logits = self.model(**bert_batch)

		output_logits = self.get_logits_fn(output_logits)

		if self.logits_transform is not None:

			batch_size = output_logits.size(0)

			transform = torch.stack([self.logits_transform for _ in range(batch_size)])

			output_logits = torch.gather(output_logits,dim=1,index=transform)

		state_object[self.output_key] = output_logits

		return state_object