# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10      #
#       Created on: 2023-01-17      #
# --------------------------------- #


import numpy as np
import torch
import time


class NDDetailedInferenceEvaluator:

	def __init__(self, *args, **kwargs):

		self.time_key = kwargs.pop("time_key")
		self.epochs = kwargs.pop("epochs")
		self.example_numbers_key = kwargs.pop("example_numbers_key")
		self.class_predictions_key = kwargs.pop("class_predictions_key")
		self.class_percentages_key = kwargs.pop("class_percentages_key")

		self.dataloader = None
		self.metric_tracker = None


	def set_dataloader(self, dataloader):
		self.dataloader = dataloader

	def set_metric_tracker(self, metric_tracker):
		self.metric_tracker = metric_tracker


	def __call__(self, model):

		self.metric_tracker.reset_metrics()	

		#make sure we are in evaluation mode and not tracking gradients
		model.eval()

		with torch.no_grad():	

			metrics_list = []
			details_list = []

			for _ in range(self.epochs):

				start_time = time.time()

				prediction_details = {}

				# Iterate through all the batches
				for batch_data in self.dataloader:

					state_object = model(batch_data)
					self.metric_tracker.update_metrics(state_object)		

					example_numbers = state_object[self.example_numbers_key]
					class_predictions = state_object[self.class_predictions_key].cpu().numpy()
					class_percentages = state_object[self.class_percentages_key].cpu().numpy()

					for num, pred, probs in zip(example_numbers, class_predictions, class_percentages):
						prediction_details[num] = (pred, probs)

				inference_metrics = self.metric_tracker.get_metrics()

				inference_metrics[self.time_key] = (time.time() - start_time)/60.0

				metrics_list.append(inference_metrics)
				details_list.append(prediction_details)

			metric_keys = list(metrics_list[0].keys())
			metrics = {key:[] for key in metric_keys}

			for batch_metrics in metrics_list:

				for key, value in batch_metrics.items():
					metrics[key].append(value)

			metrics = {key:(sum(value)/len(value)) for key, value in metrics.items()}

			example_keys = list(details_list[0].keys())
			details = {key:[] for key in example_keys}

			for batch_details in details_list:

				for key, (_, probs) in batch_details.items():
					details[key].append(probs)

			details = {key: np.stack(value,axis=0) for key, value in details.items()}

			details = {key:np.mean(value, axis=0) for key, value in details.items()}

			details = {key:(np.argmax(value),value) for key, value in details.items()}
		
			return metrics, details
