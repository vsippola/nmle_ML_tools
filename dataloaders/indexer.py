# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10       #
#       Created on: 2023-01-15      #
# --------------------------------- #


from torch.utils.data import Dataset
import torch



class IndexerDataset(Dataset):

	def __init__(self, *args, **kwargs):
		super(IndexerDataset, self).__init__()	


	def set_corpus(self, corpus):
		self.examples = corpus		


	def set_transform_fn(self, transform_fn):
		self.transform_fn = transform_fn


	def set_collate_fn(self, collate_fn):
		self.collate_fn = collate_fn

		
	def __len__(self):
		return len(self.examples)


	def __getitem__(self, index):
		return self.transform_fn(self.examples[index])
