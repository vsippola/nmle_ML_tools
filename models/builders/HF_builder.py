# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-03-10      #
# --------------------------------- #

import sys

from .context import definition
from definition.HF_model import HFModel

from transformers import RobertaAdapterModel, AutoModelForSequenceClassification, BartForSequenceClassification


class HFModelBuilder():

	model_classes = {
		"auto_seq_class":AutoModelForSequenceClassification,
		"bart_seq_class":BartForSequenceClassification,
		"roberta_adapter":RobertaAdapterModel
	}
	
	def __init__(self, *args, **kwargs):	

		self.configured = False


	def configure(self, *args, **kwargs):

		model_class = kwargs.pop("class")
		model_address = kwargs.pop("address")
		if model_class == "roberta_adapter":
			adapter_name = kwargs.pop("adapter")

		self.configured = True
		
		self.model_class = HFModelBuilder.model_classes[model_class]
		self.model_address = model_address
		self.adapter_name = adapter_name if model_class == "roberta_adapter" else None
		self.RA_params = kwargs		


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		hf_model = HFModel(**self.RA_params)

		model = self.model_class.from_pretrained(self.model_address)

		if self.adapter_name is not None:
			adapter_name = model.load_adapter(self.adapter_name, source="hf")
			model.active_adapters = adapter_name

		hf_model.set_model(model)
		
		if self.model_class == BartForSequenceClassification:

			hf_model.set_get_logits_fn(self.key_get_logits_fn(0))

		else:

			hf_model.set_get_logits_fn(self.dot_logtits_get_logits_fn())
		

		return hf_model


	def dot_logtits_get_logits_fn(self):

		def get_logits_fn(output):

			return output.logits

		return get_logits_fn


	def key_get_logits_fn(self, key):

		def get_logits_fn(output):

			return output[key]

		return get_logits_fn