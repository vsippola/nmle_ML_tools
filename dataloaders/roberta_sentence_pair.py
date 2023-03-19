# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10       #
#       Created on: 2023-01-15      #
# --------------------------------- #


from torch.utils.data import Dataset
import torch



class RobertaSentencePairDataset(Dataset):

	def __init__(self, *args, **kwargs):

		super(RobertaSentencePairDataset, self).__init__()

		indexes_key = kwargs.pop("indexes_key")
		labels_key = kwargs.pop("labels_key")
		bert_batch_key = kwargs.pop("bert_batch_key") 
		padding_index = kwargs.pop("padding_index") 


		self.collate_fn = RobertaSentencePairDataset._get_coallate_fn(indexes_key, labels_key, bert_batch_key, padding_index)


	def set_corpus(self, corpus):

		self.examples = corpus		
		
	def __len__(self):
		
		return len(self.examples)


	def __getitem__(self, index):
		
		return self.examples[index] 


	@staticmethod
	def _get_coallate_fn(indexes_key, labels_key, bert_batch_key, padding_index):

		def _collate_fn(data):	

			with torch.no_grad():

				indexes, labels, word_indexes, attention_values = zip(*data)

				#labels needs to be tensor
				labels = torch.tensor(labels)

				word_indexes = torch.nn.utils.rnn.pad_sequence(word_indexes, batch_first=True, padding_value=padding_index)
				attention_values = torch.nn.utils.rnn.pad_sequence(attention_values, batch_first=True, padding_value=0)

				state_object = {
					indexes_key: indexes,
					labels_key: labels,
					bert_batch_key:
					{
						"input_ids":word_indexes,
						"attention_mask":attention_values
					}

				}


				return state_object

		return _collate_fn
