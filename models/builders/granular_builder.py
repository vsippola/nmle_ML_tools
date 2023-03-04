# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

import sys

from .context import definition
from definition.granular_model import GranularModel



class GranularModelBuilder():
	
	def __init__(self, *args, **kwargs):

		self.configured = False


	def configure(self, *args, **kwargs):

		if "submodule_configs" not in kwargs:
			print()
			print("no submodules configurations provided")
			sys.exit()

		self.configured = True
		self.granular_params = {"input_dimension":kwargs.pop("input_dimension")}
		self.submodule_configs = kwargs.pop("submodule_configs")
			
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

		granular_model = GranularModel(**self.granular_params)


		prev_output_dimension = granular_model.input_dimension

		for config in self.submodule_configs:			

			config["input_dimension"] = prev_output_dimension

			submodule = ModuleFactory.BUILD_MODULE(**config)

			prev_output_dimension = submodule.output_dimension

			granular_model.nn_block_list.append(submodule)

		granular_model.input_dimension = granular_model.nn_block_list[0].input_dimension	
		granular_model.output_dimension = granular_model.nn_block_list[-1].output_dimension	

		granular_model.set_sequential()

		return granular_model





		

					


