# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Chahna Dixit        #
#       Python Version: 3.7.5       #
#       Created on: 2021-01-04      #
#       Modified: 2021-06-24        #
#       by Anemily Sippola          #
# --------------------------------- #


import torch
import torch.nn as nn
import inspect
import re
from torch import optim

from .custom_nn_module import CustomNNModule

class Optimizer(CustomNNModule):

	"""Initializes loss, optimizer and scheduler"""
	def __init__(self, *args, **kwargs):
		super(Optimizer, self).__init__(*args, **kwargs)

		self.optimizer_fn = None
		self.optimizer_params = None
		self.optimizer = None
		
		self.scheduler_fn = None
		self.scheduler_params = None
		self.scheduler = None


	def set_optimizer(self, optimizer_fn, optimizer_params):
		self.optimizer_fn = optimizer_fn
		self.optimizer_params = optimizer_params


	def set_scheduler(self, scheduler_fn, scheduler_params):
		self.scheduler_fn = scheduler_fn
		self.scheduler_params = scheduler_params


	def register_model(self, model):
		if self.optimizer_fn is not None:
			self.optimizer = self.optimizer_fn(model.parameters(), **self.optimizer_params)

		if (self.optimizer is not None) and (self.scheduler_fn is not None):
			self.scheduler = self.scheduler_fn(self.optimizer, **self.scheduler_params)


	def zero_grad(self, set_to_none=False):

		if self.optimizer is not None:
			self.optimizer.zero_grad(set_to_none)


	def optimizer_step(self):

		if self.optimizer is not None:
			self.optimizer.step()


	def scheduler_step(self):

		if self.scheduler is not None:
			self.scheduler.step()

	def current_lr(self):

		if self.scheduler is None:
			return None

		return self.scheduler.get_last_lr()[0]

	
	def get_param_dict(self):

		param_dict={
			"optimizer":self.optimizer.state_dict() if self.optimizer is not None else "",
			"scheduler":self.scheduler.state_dict() if self.scheduler is not None else ""
		}

		return param_dict


	def load_from_dict(self, optimizer, scheduler):

		if self.optimizer is not None:
			self.optimizer.load_state_dict(optimizer)
		if self.scheduler is not None:
			self.scheduler.load_state_dict(scheduler)
