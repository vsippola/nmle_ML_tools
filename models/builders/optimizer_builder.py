# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10      #
#       Created on: 2023-01-28      #
# --------------------------------- #


from torch import optim

from .context import definition
from definition.optimizer import Optimizer

class OptimizerBuilder():

	optimizer_fns = {
		"adadelta":optim.Adadelta, 
		"adagrad":optim.Adagrad,
		"adam":optim.Adam,
		"adamw":optim.AdamW,
		"sparseadam":optim.SparseAdam,
		"adamax":"optim.Adamax",
		"asgd":optim.ASGD,
		"rmsprop":optim.RMSprop,
		"rprop":optim.Rprop,
		"sgd":optim.SGD
		}

	scheduler_fns = {
		"steplr":optim.lr_scheduler.StepLR
	}

	"""Initializes loss, optimizer and scheduler"""
	def __init__(self):

		self.configured = False


	def configure(self, *args, **kwargs):

		optimizer_params = kwargs.pop("optimizer_params", None)
		scheduler_params = kwargs.pop("scheduler_params", None)

		self.configured = True

		self.optimizer_params = optimizer_params
		self.scheduler_params = scheduler_params


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		optimizer = Optimizer()

		if (self.optimizer_params is not None) and (self.optimizer_params["optimizer_type"] is not None):

			optimizer_type = self.optimizer_params.pop("optimizer_type")
			optimzer_fn = OptimizerBuilder.optimizer_fns[optimizer_type]

		else:

			optimzer_fn = None

		optimizer.set_optimizer(optimzer_fn, self.optimizer_params)


		if (self.scheduler_params is not None) and (self.scheduler_params["scheduler_type"] is not None):
			
			scheduler_type = self.scheduler_params.pop("scheduler_type")
			scheduler_fn = OptimizerBuilder.scheduler_fns[scheduler_type]

		else:

			scheduler_fn = None

		optimizer.set_scheduler(scheduler_fn, self.scheduler_params)

		return optimizer



		


