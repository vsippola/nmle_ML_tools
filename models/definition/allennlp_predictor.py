# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-03-10      #
# --------------------------------- #

from .custom_nn_module import CustomNNModule

import numpy as np
import torch


class AllenNLPPredictorModel(CustomNNModule):
	
	def __init__(self, *args, **kwargs):	

		self.json_batch_key = kwargs.pop("json_batch_key")	
		self.output_key = kwargs.pop("output_key")			
		self.output_dimension = kwargs.pop("output_dimension")	
		self.output_transform = kwargs.pop("output_transform", range(self.output_dimension))

		self.input_dimension = 1
		
		super(AllenNLPPredictorModel, self).__init__(*args, **kwargs)	

		self.model = None


	def set_model(self, model):		

		self.model = model
		

	def forward(self, state_object):

		json_batch = state_object.pop(self.json_batch_key)	

		preds = self.model.predict_batch_json(json_batch)	

		logits = [[pred["label_logits"][p_i] for p_i in self.output_transform] for pred in preds]

		logits = torch.tensor(logits, device=self.device)

		state_object[self.output_key] = logits

		return state_object