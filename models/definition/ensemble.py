# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-01-31      #
# --------------------------------- #

import torch
from .custom_nn_module import CustomNNModule

class EnsembleBlock(CustomNNModule):
	
	def __init__(self, *args, **kwargs):				

		self.input_dimension = kwargs.pop("input_dimension")
		self.ensemble_values_key = kwargs.pop("ensemble_values_key")
		self.output_key = kwargs.pop("output_key")
		self.class_percentages_key = kwargs.pop("class_percentages_key")

		#overwrite defaults
		kwargs["freeze"] = kwargs.pop("freeze", True)
		
		super(EnsembleBlock, self).__init__(*args, **kwargs)

		self.ensemble_weights = None
		self.mask = None
		self.noise_generator_param = None
		self.output_dimension = self.input_dimension

	
	def set_ensemble_weights(self, ensemble_weights):			
		self.ensemble_weights = torch.nn.ParameterList([torch.nn.Parameter(w, requires_grad=not self._freeze) for w in ensemble_weights])


	def set_mask(self, mask):
		self.mask = mask
		self.mask = [ (mul_mask.to(self.device), add_mask.to(self.device)) for mul_mask, add_mask in zip(*self.mask)]


	def set_noise_generator_param(self, noise_generator_param):
		self.noise_generator_param = noise_generator_param


	def to_device(self):
		self.to(self.device)

		if self.mask is not None:
			self.mask = [ (mul_mask.to(self.device), add_mask.to(self.device)) for mul_mask, add_mask in self.mask]

		if self.noise_generator_param is not None:
			for p_i, param in enumerate(self.noise_generator_param):
				self.noise_generator_param[p_i] = param.to(self.device)


	def _get_weights(self, batch_size):

		#apply masks
		if self.mask is not None:
			ensemble_weights = [torch.mul(weights, mul_mask) + add_mask for weights, (mul_mask, add_mask) in zip(self.ensemble_weights, self.mask)]
		else:
			ensemble_weights = self.ensemble_weights

		#apply soft max
		softmax_weights = [torch.nn.functional.softmax(w, dim=0)for w in ensemble_weights]

		indexes = torch.randint(low=0, high=len(self.ensemble_weights), size=(batch_size,))

		weights = [softmax_weights[i] for i in indexes]

		weights = torch.stack(weights)

		if self.noise_generator_param is not None:

			mean, std = self.noise_generator_param
			noise_generator = torch.distributions.normal.Normal(mean, std)

			noise = noise_generator.sample(weights.size()).squeeze()

			if self.mask is not None:

				noise_mask = torch.stack([self.mask[i][0] for i in indexes])
				noise = torch.mul(noise, noise_mask)

			noisy_weights = weights + noise

			noisy_weights = torch.nn.functional.relu(noisy_weights)

			noisy_weights_sum = torch.sum(noisy_weights, dim=1).unsqueeze(1)

			weights = noisy_weights/noisy_weights_sum
	
		weights = weights.unsqueeze(2)

		return weights
		

	def forward(self, state_object):

		ensemble_values = state_object[self.ensemble_values_key]
		ensemble_values = ensemble_values.to(self.device)

		batch_size = ensemble_values.size(0)

		weights = self._get_weights(batch_size)

		weighted_values = torch.mul(ensemble_values, weights)

		output_logits = torch.sum(weighted_values, dim=1)

		state_object[self.output_key] = output_logits
		state_object[self.class_percentages_key] = output_logits

		return state_object
