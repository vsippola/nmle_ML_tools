# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10      #
#       Created on: 2023-01-17      #
# --------------------------------- #


import torch
import time

class DetailedInferenceEvaluator:

	def __init__(self, *args, **kwargs):

		self.time_key = kwargs.pop("time_key")
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

		start_time = time.time()

		self.metric_tracker.reset_metrics()	

		#make sure we are in evaluation mode and not tracking gradients
		model.eval()

		prediction_details = {}

		with torch.no_grad():		

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
		
			return inference_metrics, prediction_details
