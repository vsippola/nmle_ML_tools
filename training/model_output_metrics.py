# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10       #
#       Created on: 2023-01-15      #
# --------------------------------- #

import torch

class ModelPredictionMetrics():

	def __init__(self, *arg, **kwargs):

		self.num_classes = kwargs.pop("number_of_classes")
		self.model_output_keys = kwargs.pop("model_output_keys")
		self.metric_output_keys = kwargs.pop("metric_output_keys")

		self.reset_metrics()


	def reset_metrics(self):

		self.average_batch_loss = []
		self.confusion_matrix = [[0.0 for _ in range(self.num_classes)] for _ in range(self.num_classes)]


	def update_metrics(self, state_object):

		with torch.no_grad():

			batch_loss = state_object.get(self.model_output_keys["batch_loss"], None)
			true_labels = state_object.get(self.model_output_keys["true_labels"], None)
			class_predictions = state_object.get(self.model_output_keys["class_predictions"], None)

			if batch_loss is not None:
				self.average_batch_loss.append(batch_loss.detach().cpu().numpy())				

			if (true_labels is not None) and (class_predictions is not None):
				
				#I don't think you need to do this, but jic
				true_labels = true_labels.detach().cpu()
				class_predictions = class_predictions.detach().cpu()

				for true_label, class_pred in zip(true_labels, class_predictions):
					self.confusion_matrix[true_label][class_pred] += 1


	def get_average_batch_loss(self):

		with torch.no_grad():

			return sum(self.average_batch_loss)/len(self.average_batch_loss)


	def get_confusion_matrix(self):

		return self.confusion_matrix


	#add more as we want them, this should be all we need for training
	def get_metrics(self):

		with torch.no_grad():

			confusion_matrix = self.confusion_matrix
			num_classes = self.num_classes

			total_preds = sum([sum(row) for row in confusion_matrix])

			correct_preds = sum([confusion_matrix[i][i] for i in range(num_classes)])			

			accuracy = correct_preds/total_preds
			loss = self.get_average_batch_loss()

			metrics = {				
				self.metric_output_keys["loss"]: loss,
				self.metric_output_keys["accuracy"]: accuracy
			}

		return metrics



		







