# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina        #
#       Python Version: 3.10       #
#       Created on: 2023-02-16      #
# --------------------------------- #


from .custom_nn_module import CustomNNModule

class LossBlock(CustomNNModule):

	"""Initializes loss, optimizer and scheduler"""
	def __init__(self, *args, **kwargs):		
		
		super(LossBlock, self).__init__(device)
		
		self.prediction_logits_key = kwargs.pop("prediction_logits_key")
		self.true_labels_key = kwargs.pop("true_labels_key")

		self.loss_key = kwargs.pop("loss_key")

		self.loss_fn = None


	def set_loss_fn(self, fn):

		self.loss_fn = fn


	def forward(self, state_object):

		prediction_logits = state_object[self.prediction_logits_key]
		prediction_logits = prediction_logits.to(self.device)

		true_labels = state_object[self.true_labels_key]
		true_labels = true_labels.to(self.device)

		loss = self.loss_fn(prediction_logits, true_labels)

		state_object[self.loss_key] = loss

		return state_object
	

	#def to_device(self):
	#	self.loss_fn.to(self.device)		


