# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10       #
#       Created on: 2023-03-15      #
# --------------------------------- #


import torch


class CollateFunctions():
	
	def collate_stack_sentences(*args, **kwargs):

		example_numbers_key = kwargs.pop("example_numbers_key")
		labels_key = kwargs.pop("labels_key")
		word_indexes_key = kwargs.pop("word_indexes_key")
		sentence_lengths_key = kwargs.pop("sentence_lengths_key")

		def collate_fn(data):

			with torch.no_grad():

				example_numbers, labels, sents = zip(*data)

				#labels needs to be tensor
				labels = torch.tensor(labels)

				sentences = []
				for s_list in zip(*sents):
					sentences += s_list

				#get lengths for padding/packing
				sentences_lenghts = [len(s) for s in sentences]

				#flatten word indexes and make them a tensor
				word_indexes = [w_i for s in sentences for w_i in s]
				word_indexes = torch.tensor(word_indexes)

				state_object = {
					example_numbers_key: example_numbers,
					labels_key: labels,
					word_indexes_key: word_indexes,
					sentence_lengths_key: sentences_lenghts
				}

				return state_object

		return collate_fn


	COLLATE_FN_DICT = {
		"stack_sentences":collate_stack_sentences
	}
	

	def get_fn(*args, **kwargs):

		collate_fn_type = kwargs.pop('collate_fn_type')
		collate_fn = CollateFunctions.COLLATE_FN_DICT[collate_fn_type](**kwargs)

		return collate_fn

	
