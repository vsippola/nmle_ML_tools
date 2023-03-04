# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10       #
#       Created on: 2023-01-15      #
# --------------------------------- #

from torch.utils.data import Dataset
import torch

class SentencePairDataset(Dataset):

	def __init__(self, *args, **kwargs):

		super(SentencePairDataset, self).__init__()

		indexes_key = kwargs.pop("indexes_key")
		labels_key = kwargs.pop("labels_key")
		word_indexes_key = kwargs.pop("word_indexes_key") #this is misnamed, it just needs to be a per word identifier
		sentence_lengths_key = kwargs.pop("sentence_lengths_key")

		self.collate_fn = SentencePairDataset._get_coallate_fn(indexes_key, labels_key, word_indexes_key, sentence_lengths_key)


	def set_corpus(self, corpus):

		self.examples = corpus		

		
	def __len__(self):
		
		return len(self.examples)


	def __getitem__(self, index):
		
		return self.examples[index] 


	@staticmethod
	def _get_coallate_fn(indexes_key, labels_key, word_indexes_key, sentence_lengths_key):

		def _collate_fn(data):

			"""
			(indexes, labels, sents1, sents2)
			"""

			with torch.no_grad():

				indexes, labels, sents1, sents2 = zip(*data)

				#labels needs to be tensor
				labels = torch.tensor(labels)

				#stack sentences so [right sides] + [left sides]
				sentences = sents1 + sents2

				#get lengths for padding/packing
				sentences_lenghts = [len(s) for s in sentences]

				#flatten word indexes and make them a tensor
				word_indexes = [w_i for s in sentences for w_i in s]
				word_indexes = torch.tensor(word_indexes)

				state_object = {
					indexes_key: indexes,
					labels_key: labels,
					word_indexes_key: word_indexes,
					sentence_lengths_key: sentences_lenghts
				}

				return state_object

		return _collate_fn
