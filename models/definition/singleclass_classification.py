# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-01-31      #
# --------------------------------- #

"""
Essentially just wraps an actual model and outputs the loss and predictions
"""

import torch

from .custom_nn_module import CustomNNModule


class SingleClassClassification(CustomNNModule):
	
	def __init__(self, *args, **kwargs):				

		self.number_of_classes = kwargs.pop("number_of_classes")	

		self.prediction_logits_key = kwargs.pop("prediction_logits_key")	
		self.true_labels_key = kwargs.pop("true_labels_key")	

		self.loss_values_key = kwargs.pop("loss_values_key", None)
		self.class_predictions_key = kwargs.pop("class_predictions_key")
		self.class_percentages_key = kwargs.pop("class_percentages_key", None)
		
		#call to custom NN module
		super(SingleClassClassification, self).__init__(*args, **kwargs)

		self.model = None
		self.loss_fn = None
		self.predict_class = None
		self.output_dimension = None

	
	def set_model(self, model):

		self.model = model


	def set_loss_fn(self, fn):

		self.loss_fn = fn


	def set_predict_class_fn(self, fn):

		self.predict_class = fn


	def forward(self, state_object):

		state_object = self.model(state_object)

		prediction_logits = state_object[self.prediction_logits_key]

		with torch.no_grad():

			state_object[self.class_predictions_key] = self.predict_class(self, prediction_logits)

			if self.class_percentages_key is not None:

				state_object[self.class_percentages_key] = torch.nn.functional.softmax(prediction_logits, dim=1)

		if (self.loss_values_key is not None) and (self.true_labels_key in state_object):			
			
			true_labels = state_object[self.true_labels_key]
			true_labels = true_labels.to(prediction_logits.device)

			state_object[self.loss_values_key] = self.loss_fn(prediction_logits, true_labels)		

		return state_object


	def _max_value_prediction(self, predictions):

		predictions = predictions.detach()

		return (predictions.max(1)[1]).detach()  


	def get_param_dict(self):

		param_dict={
			"model_dict":self.model.get_param_dict()
		}
		return param_dict


	def load_from_dict(self, model_dict):

		self.model.load_from_dict(**model_dict)



		





