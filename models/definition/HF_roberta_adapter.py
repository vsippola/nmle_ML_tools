# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-03-10      #
# --------------------------------- #

from .custom_nn_module import CustomNNModule


class HFRobertaAdapterModel(CustomNNModule):
	
	def __init__(self, *args, **kwargs):	

		self.bert_batch_key = kwargs.pop("bert_batch_key")	
		self.output_key = kwargs.pop("output_key")	

		self.input_dimension = 1
		
		super(HFRobertaAdapterModel, self).__init__(*args, **kwargs)	
		
		self.output_dimension = None

		self.model = None


	def set_model(self, model):		

		adapter_key = list(model.config.prediction_heads.keys())[0]		
		self.output_dimension = model.config.prediction_heads[adapter_key]["num_labels"]

		self.model = model
			

	def forward(self, state_object):

		bert_batch = state_object.pop(self.bert_batch_key)
		bert_batch['input_ids'] = bert_batch['input_ids'].to(self.device)
		bert_batch['attention_mask'] = bert_batch['attention_mask'].to(self.device)

		state_object[self.output_key] = self.model(**bert_batch).logits		

		return state_object