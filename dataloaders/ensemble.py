# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10       #
#       Created on: 2023-01-15      #
# --------------------------------- #


from torch.utils.data import Dataset
import torch



class EnsembleDataset(Dataset):

	def __init__(self, *args, **kwargs):
		super(EnsembleDataset, self).__init__()	


	def set_collate_fn(self, collate_fn):
		self.collate_fn = collate_fn


	def set_corpus(self, corpus):
		self.corpus = corpus


	def set_datasets(self, datasets):
		self.datasets = datasets		


	def set_transform_fn(self, transform_fn):
		self.transform_fn = transform_fn

		
	def __len__(self):
		return len(next(iter( self.datasets.values() )))


	def __getitem__(self, index):

		example_number, label = self.transform_fn(self.corpus[index])

		return [example_number, label, [self.datasets[dataset_key][index] for dataset_key in self.datasets]]
