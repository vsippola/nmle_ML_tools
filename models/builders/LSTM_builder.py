# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

import sys

from .context import definition
from definition.LSTM import LSTMBlock



class LSTMBuilder():
	
	def __init__(self, *args, **kwargs):
		
		self.configured = False


	def configure(self, *args, **kwargs):


		self.configured = True
		
		self.LSTM_params = kwargs		
			


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		LSTM_block = LSTMBlock(**self.LSTM_params)

		return LSTM_block






		

					


