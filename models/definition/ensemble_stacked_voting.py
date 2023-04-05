# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.10.4       #
#       Created on: 2023-01-31      #
# --------------------------------- #

import numpy as np
import torch

from .custom_nn_module import CustomNNModule

class EnsembleStackedVotingBlock(CustomNNModule):
	
	def __init__(self, *args, **kwargs):				

		self.input_dimension = kwargs.pop("input_dimension")
		self.ensemble_values_key = kwargs.pop("ensemble_values_key")
		self.output_key = kwargs.pop("output_key")
		self.class_percentages_key = kwargs.pop("class_percentages_key")
	
		super(EnsembleStackedVotingBlock, self).__init__(*args, **kwargs)

		self.output_dimension = self.input_dimension


	def forward(self, state_object):

		ensemble_values = state_object[self.ensemble_values_key]
		ensemble_values = ensemble_values.to(self.device)

		scores, votes = torch.max(ensemble_values, dim=2)

		bins = range(self.output_dimension+1)
		voting_breakdown = [np.histogram(v, bins=bins)[0] for v in votes.cpu().numpy()]
		winning_size = [max(v) for v in voting_breakdown]

		winning_labels = [[label for label,num_votes  in enumerate(v) if num_votes==win_s] for v, win_s in zip(voting_breakdown, winning_size)]

		winning_voters = [[[voter for voter, vote in enumerate(v) if vote==win_l] for win_l in winning_label] for v, winning_label in zip(votes, winning_labels)]

		for (wv_i, w_voters), s in zip(enumerate(winning_voters), scores):

			while len(winning_voters[wv_i]) > 1:

				highest_score, highest_voter = torch.max(s, dim=0)

				for voters in w_voters:
					if highest_voter in voters:
						winning_voters[wv_i] = [voters]
						break

				s[highest_voter] = -1e9

			winning_voters[wv_i]=winning_voters[wv_i][0]

		output_logits = [torch.mean(values[voters], dim=0) for voters, values in zip(winning_voters, ensemble_values)]

		output_logits = torch.stack(output_logits)	
		
		state_object[self.output_key] = output_logits
		state_object[self.class_percentages_key] = output_logits

		return state_object
