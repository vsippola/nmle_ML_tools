# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #


import os
import sys

from torch.utils.data import DataLoader

from .context import dataloaders
from .context import models
from dataloaders.dataset_factory import DatasetFactory
from models.builders.optimizer_builder import OptimizerBuilder

from .display_training_results import DisplayTrainingResults
from .inference_evaluator_builder import InferenceEvaluatorBuilder
from .model_output_metrics import ModelPredictionMetrics
from .model_trainer import ModelTrainer


class ModelTrainerBuilder():
	
	def __init__(self, *args, **kwargs):
		
		self.configured = False


	def configure(self, *args, **kwargs):

		#update configuration		
		number_of_classes = kwargs.pop("number_of_classes")
		model_output_keys = kwargs.pop("model_output_keys")
		dataset_config = kwargs.pop("dataset_config")
		dataloader_params = kwargs.pop("dataloader_params")
		display_config = kwargs.pop("display_config")
		inference_evaluator_config = kwargs.pop("inference_evaluator_config")
		optimizer_config = kwargs.pop("optimizer_config")

		self.configured = True
		
		self.number_of_classes = number_of_classes
		self.model_output_keys = model_output_keys
		self.dataset_config = dataset_config
		self.dataloader_params = dataloader_params
		self.display_config = display_config
		self.inference_evaluator_config = inference_evaluator_config
		self.optimizer_config = optimizer_config
		self.model_trainer_params = kwargs


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		inference_keys = {"loss":"inference_loss", "acc":"inference_acc", "time":"inference_time"}
		training_keys = {"loss":"train_loss", "checkpoint":"checkpoint", "acc":"train_acc", "time":"train_time", "total_batches":"total_batches", "batch_idx":"batch_idx","epoch_num":"epoch_num"}

		#build data loader
		dataset = DatasetFactory.BUILD_DATASET(**self.dataset_config)
		if dataset.collate_fn is not None:
			self.dataloader_params["collate_fn"] = dataset.collate_fn
		dataloader = DataLoader(dataset, **self.dataloader_params)

		#build metric tracker
		metric_tracker_parmas = {
			"number_of_classes":self.number_of_classes,
			"model_output_keys":self.model_output_keys,
			"metric_output_keys":{
				"loss":training_keys["loss"],
				"accuracy":training_keys["acc"]
			}
		}		

		metric_tracker = ModelPredictionMetrics(**metric_tracker_parmas)

		#build the displaying object
		self.display_config["display_keys"] = {
			"epoch_num":training_keys["epoch_num"],
			"checkpoint":training_keys["checkpoint"],
			"batch_idx":training_keys["batch_idx"],
			"total_batches":training_keys["total_batches"],
			"train_loss":training_keys["loss"],
			"train_acc":training_keys["acc"],
			"train_time":training_keys["time"],
			"inference_loss":inference_keys["loss"],
			"inference_acc":inference_keys["acc"],
			"inference_time":inference_keys["time"]
		}

		training_displayer = DisplayTrainingResults(**self.display_config)

		#build the inferenece evaluator
		self.inference_evaluator_config["metric_tracker_params"] = {
			"number_of_classes":self.number_of_classes,
			"model_output_keys":self.model_output_keys,
			"metric_output_keys":{
				"loss":inference_keys["loss"],
				"accuracy":inference_keys["acc"]
			}
		}
		self.inference_evaluator_config["time_key"] = inference_keys["time"]
		inference_evaluator_builder = InferenceEvaluatorBuilder()
		inference_evaluator_builder.configure(**self.inference_evaluator_config)
		inference_evaluator = inference_evaluator_builder.build()	

		#builder optimizer
		optimzer_builder = OptimizerBuilder()
		optimzer_builder.configure(**self.optimizer_config)
		optimizer = optimzer_builder.build()

		#create instance of model trainer
		self.model_trainer_params["training_display_keys"] = training_keys
		self.model_trainer_params["inference_acc_key"] = inference_keys["acc"]		

		model_trainer = ModelTrainer(**self.model_trainer_params)
		model_trainer.set_dataloader(dataloader)
		model_trainer.set_metric_tracker(metric_tracker)
		model_trainer.set_training_displayer(training_displayer)
		model_trainer.set_inference_evaluator(inference_evaluator)
		model_trainer.set_optimizer(optimizer)

		return model_trainer
