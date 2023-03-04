# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

"""
This class builds a word2vec NN object.10

It loads vectors from a given pkl file.

"""

import sys

from torch.utils.data import DataLoader

from .context import dataloaders
from dataloaders.dataset_factory import DatasetFactory

from .inference_evaluator import InferenceEvaluator
from .model_output_metrics import ModelPredictionMetrics




class InferenceEvaluatorBuilder():
	
	def __init__(self, *args, **kwargs):
		
		self.configured = False


	def configure(self, *args, **kwargs):

		#update configuration
		dataset_config = kwargs.pop("dataset_config")
		dataloader_params = kwargs.pop("dataloader_params")
		metric_tracker_params = kwargs.pop("metric_tracker_params")

		self.configured = True
		
		self.dataset_config = dataset_config
		self.dataloader_params = dataloader_params			
		self.metric_tracker_params = metric_tracker_params	
		self.inference_evaluator_params = kwargs


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		dataset = DatasetFactory.BUILD_DATASET(**self.dataset_config)
		self.dataloader_params["collate_fn"] = dataset.collate_fn
		dataloader = DataLoader(dataset, **self.dataloader_params)

		metric_tracker = ModelPredictionMetrics(**self.metric_tracker_params)

		inference_evaluator = InferenceEvaluator(**self.inference_evaluator_params)		
		inference_evaluator.set_dataloader(dataloader)
		inference_evaluator.set_metric_tracker(metric_tracker)

		return inference_evaluator
