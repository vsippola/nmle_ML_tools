# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

import torch
import sys

from .context import definition
from definition.singleclass_classification import SingleClassClassification



class SingleClassClassificationBuilder():

	def negative_log():

		def loss_fn(predictions, true_labels):

			predictions = torch.stack([p[tl_i] for p, tl_i in zip(predictions, true_labels)])

			predictions = -torch.log(predictions)

			predictions = torch.mean(predictions)

			return predictions

		return loss_fn


	LOSS_FNS = {
		"cross_entropy":torch.nn.CrossEntropyLoss,
		"neg_log":negative_log
	}

	PREDICTION_FNS = {"max": SingleClassClassification._max_value_prediction}
	
	def __init__(self, *args, **kwargs):

		self.configured = False


	def configure(self, *args, **kwargs):		

		loss_fn_param = kwargs.pop("loss_fn")
		prediction_fn_loss_fn_param = kwargs.pop("prediction_fn")
		module_config = kwargs.pop("module_config")

		self.configured = True
		self.loss_fn_param = loss_fn_param
		self.prediction_fn_param = prediction_fn_loss_fn_param
		self.module_config = module_config
		self.SCC_params = kwargs
			
	"""
	This function should probably do a forward pass on the input/output keys and throw an error if there is ever a key missing
	However this requires changing the way input/outut keys are used

	TODO do this
	"""
	def build(self):

		#this is here to avoid a circular dependancy?
		from models.builders.module_factory import ModuleFactory

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		singleclass_model = SingleClassClassification(**self.SCC_params)		

		loss_fn_type = self.loss_fn_param.pop("loss_fn_type")
		loss_fn = SingleClassClassificationBuilder.LOSS_FNS[loss_fn_type](**self.loss_fn_param)

		pred_fn_type = self.prediction_fn_param.pop("prediction_fn_type")
		pred_fn = SingleClassClassificationBuilder.PREDICTION_FNS[pred_fn_type]

		submodule = ModuleFactory.BUILD_MODULE(**self.module_config)

		singleclass_model.set_model(submodule)
		singleclass_model.output_dimension = submodule.output_dimension
		singleclass_model.set_loss_fn(loss_fn)
		singleclass_model.set_predict_class_fn(pred_fn)

		return singleclass_model





		

					


