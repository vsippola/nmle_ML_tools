# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10       #
#       Created on: 2023-03-15      #
# --------------------------------- #

import os
import pickle

class SentenceTransformFunctions():
	
	def sentence_transform_w2v(*args, **kwargs):

		from nltk.tokenize import word_tokenize

		word2idx_file = kwargs.pop("word2idx_file")

		if not (os.path.isfile(word2idx_file)):
			print()
			print(f'file {word2idx_file} does not exist')
			sys.exit()	

		with open(word2idx_file, "rb") as f:
			word2idx = pickle.load(f)

		sentence_index = kwargs.pop("sentence_index")

		combine_before_transform = kwargs.pop("combine_before_transform")

		if isinstance(sentence_index, list):

			if combine_before_transform:

				def sentence_transform_fn(example):

					sentences = [example[s_i] for s_i in sentence_index]
					sentence = " ".join(sentences)

					return [word2idx[word_token] if word_token in word2idx else 0 for word_token in word_tokenize(sentence)]

			else:

				def sentence_transform_fn(example):

					sentences = [example[s_i] for s_i in sentence_index]

					return [[word2idx[word_token] if word_token in word2idx else 0 for word_token in word_tokenize(sentence)] for sentence in sentences]

		else:

			def sentence_transform_fn(example):

				sentence = example[sentence_index]

				return [word2idx[word_token] if word_token in word2idx else 0 for word_token in word_tokenize(sentence)]


		return sentence_transform_fn

		

	def sentence_transform_gather(*args, **kwargs): 

		sentence_index = kwargs.pop("sentence_index")

		combine_before_transform = kwargs.pop("combine_before_transform")

		if isinstance(sentence_index, list):

			if combine_before_transform:

				combine_seperator = kwargs.pop("combine_seperator")

				def sentence_transform_fn(example):

					sentences = [example[s_i] for s_i in sentence_index]
					sentence = combine_seperator.join(sentences)

					return sentence

			else:

				def sentence_transform_fn(example):

					return  [example[s_i] for s_i in sentence_index]

		else:

			def sentence_transform_fn(example):

				return  example[sentence_index] 

		return sentence_transform_fn


	def sentence_transform_bert(*args, **kwargs):

		sentence_index = kwargs.pop("sentence_index")

		add_sep = kwargs.pop("add_sep", True)
		if add_sep:
			sep_token = kwargs.pop("sep_token", "</s>")
			

		if isinstance(sentence_index, list):

			if add_sep:

				def sentence_transform_fn(example):

					sentences = [example[s_i] for s_i in sentence_index]
					sentence = f" {sep_token} ".join(sentences)

					return sentence

				return sentence_transform_fn

		

	SENTENCE_TRANSFORM_FN_DICT = {
		"bert":sentence_transform_bert,
		"gather":sentence_transform_gather,
		"w2v":sentence_transform_w2v
	}

	def get_fn(*args, **kwargs):

		sentence_tranform_type = kwargs.pop('sentence_transform_type')
		sentence_transform_fn = SentenceTransformFunctions.SENTENCE_TRANSFORM_FN_DICT[sentence_tranform_type](**kwargs)

		return sentence_transform_fn

	
