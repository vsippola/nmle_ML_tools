# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-03-10      #
# --------------------------------- #

import sys

from .context import definition
from definition.HF_roberta_adapter import HFRobertaAdapterModel

from transformers import RobertaAdapterModel


class HFRobertaAdapterModelBuilder():
	
	def __init__(self, *args, **kwargs):	

		self.configured = False


	def configure(self, *args, **kwargs):

		roberta_base = kwargs.pop("roberta_base")
		adapter_name = kwargs.pop("adapter_name")

		self.configured = True
		
		self.roberta_base = roberta_base
		self.adapter_name = adapter_name
		self.RA_params = kwargs		
			


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		ra_model = HFRobertaAdapterModel(**self.RA_params)

		hf_model = RobertaAdapterModel.from_pretrained(self.roberta_base)
		adapter_name = hf_model.load_adapter(self.adapter_name, source="hf")
		hf_model.active_adapters = adapter_name

		ra_model.set_model(hf_model)

		return ra_model
