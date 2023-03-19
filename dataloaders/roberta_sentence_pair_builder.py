# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

import os
import pathlib
import pickle
import sys

from transformers import RobertaTokenizer

from .roberta_sentence_pair import RobertaSentencePairDataset

class RobertaSentencePairDatasetBuilder():
	
	def __init__(self, *args, **kwargs):
		
		self.configured = False
		self._CACHE_PREFIX = "cy6AHonOMo_roberta_sp_"


	def configure(self, *args, **kwargs):

		#update configuration
		corpus_file = kwargs.pop("corpus_file")
		tokenizer_source = kwargs.pop("tokenizer_source")
		label2int = kwargs.pop("label2int")
		cache_folder = kwargs.pop("cache_folder", None)

		id_idx = kwargs.pop("id_idx", 0)
		label_idx = kwargs.pop("label_idx", 1)
		s1_idx = kwargs.pop("s1_idx", 2)
		s2_idx = kwargs.pop("s2_idx", 3)
		
		if not (os.path.isfile(corpus_file)):
			print()
			print(f'file {corpus_file} does not exist')
			sys.exit()

		self.configured = True
		
		self.corpus_file = corpus_file
		self.tokenizer_source = tokenizer_source		
		self.label2int = label2int
		self.cache_folder = cache_folder

		self.id_idx = id_idx
		self.label_idx = label_idx
		self.s1_idx = s1_idx
		self.s2_idx = s2_idx

		self.dataloader_params = kwargs		


	def make_cache_filename(self):

		corpus_base_fname = os.path.splitext(os.path.basename(self.corpus_file))[0]

		cache_base_fname = f"{self._CACHE_PREFIX}_{corpus_base_fname}_{self.tokenizer_source}.pkl"
		
		cache_filename = os.path.join(self.cache_folder, cache_base_fname)

		return cache_filename


	def preprocess_corpus(self):

		print()
		print("Preprocessing corpus")
		print()


		PRINT_STEP = 2500
		line_count = 0

		corpus = []
		with open(self.corpus_file, 'r') as f:
			for example in f:

				example_tokens = example.split('\t')

				example_id = example_tokens[self.id_idx]
				example_label = example_tokens[self.label_idx]
				sents = [example_tokens[self.s1_idx], example_tokens[self.s2_idx]]

				#some examples may not have correct labels ignore them
				if example_label in self.label2int:

					example_id = int(example_id)
					example_label = self.label2int[example_label]
					concat_sent = f"{sents[0]}<sep>{sents[1]}"

					bert_input = self.tokenizer(concat_sent, return_tensors='pt')

					word_indexes = bert_input["input_ids"].squeeze()
					attention_mask = bert_input["attention_mask"].squeeze()

					example = [example_id, example_label, word_indexes, attention_mask]
					corpus.append(example)

				if (line_count+1) % PRINT_STEP == 0:
					print(f'\rlines processed: {line_count+1}', end='\r')

				line_count += 1

		print()
		print()

		return corpus


	def build(self):

		if not self.configured:
			print()
			print(f'Builder not configured')
			sys.exit()

		self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_source)
		self.dataloader_params["padding_index"] = self.tokenizer(self.tokenizer.pad_token)["input_ids"][1]

		dataset = RobertaSentencePairDataset(**self.dataloader_params)	

		#if a cache folder is set
		if self.cache_folder is not None:

			cache_filename = self.make_cache_filename()

			#if the cached file exists load it
			if os.path.isfile(cache_filename):

				print()
				print(f"Loading corpus {self.corpus_file} from cache")

				with open(cache_filename, "rb") as f:
					corpus = pickle.load(f)

				dataset.set_corpus(corpus)

				return dataset

		

		#otherwise there is no cached copy of the preprocessed corpus
		corpus = self.preprocess_corpus()
		
		#cache copy if folder exists
		if self.cache_folder is not None:

			pathlib.Path(self.cache_folder).mkdir(parents=True, exist_ok=True)

			cache_filename = self.make_cache_filename()

			with open (cache_filename, "wb") as f:
				pickle.dump(corpus, f)

		dataset.set_corpus(corpus)

		return dataset
					

