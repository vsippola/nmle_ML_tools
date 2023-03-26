# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-03-10      #
# --------------------------------- #

import sys

from .context import definition
from definition.allennlp_predictor import AllenNLPPredictorModel

from allennlp_models.pretrained import load_predictor 


class AllenNLPPredictorModelBuilder():
	
	def __init__(self, *args, **kwargs):	

		self.configured = False


	def configure(self, *args, **kwargs):

		predictor_source = kwargs.pop("predictor_source")
		
		self.configured = True
		
		self.predictor_source = predictor_source		
		self.pred_params = kwargs		
			


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		allen_model = AllenNLPPredictorModel(**self.pred_params)		

		predictor = load_predictor(self.predictor_source, cuda_device = -1 if allen_model.device == "cpu" else 0)
		
		allen_model.set_model(predictor)

		return allen_model
