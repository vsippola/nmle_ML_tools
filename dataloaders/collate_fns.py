# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10       #
#       Created on: 2023-03-15      #
# --------------------------------- #


import torch


class CollateFunctions():

	def allennlp_snli(*args, **kwargs):

		example_numbers_key = kwargs.pop("example_numbers_key", None)
		labels_key = kwargs.pop("labels_key", None)
		json_batch_key = kwargs.pop("json_batch_key")

		def collate_fn(data):

			with torch.no_grad():

				example_numbers, labels, sents = zip(*data)

				#labels needs to be tensor
				labels = torch.tensor(labels, dtype=torch.long)

				json_batch = []
				for s1, s2 in sents:
					json_example = {"premise":s1, "hypothesis":s2}
					json_batch.append(json_example)

				state_object = {}

				for key, value in zip([example_numbers_key, labels_key, json_batch_key],[example_numbers, labels, json_batch]):
					if key is not None:
						state_object[key] = value

				return state_object

		return collate_fn


	def ensemble(*args, **kwargs):

		example_numbers_key = kwargs.pop("example_numbers_key", None)
		labels_key = kwargs.pop("labels_key", None)
		collate_fns = kwargs.pop("collate_fns")
		enemble_batch_key = kwargs.pop("ensemble_batch_key")

		def collate_fn(data):

			with torch.no_grad():

				example_numbers, labels, emsemble_batches = zip(*data)

				labels = torch.tensor(labels)

				emsemble_batches = zip(*emsemble_batches)

				ensemble_batch = {}

				for (dataset_key, collate_fn), batch in zip(collate_fns.items(), emsemble_batches):

					if collate_fn is None:

						batch = torch.utils.data.default_collate(batch)

					else:

						batch = collate_fn(batch)

					ensemble_batch[dataset_key] = batch

				state_object = {}

				for key, value in zip([example_numbers_key, labels_key, enemble_batch_key],[example_numbers, labels, ensemble_batch]):
					if key is not None:
						state_object[key] = value

				return state_object

		return collate_fn


	def ensemble_values(*args, **kwargs):

		example_numbers_key = kwargs.pop("example_numbers_key", None)
		labels_key = kwargs.pop("labels_key", None)
		enemble_batch_key = kwargs.pop("ensemble_batch_key")

		def collate_fn(data):

			with torch.no_grad():

				example_numbers, labels, emsemble_batches = zip(*data)

				labels = torch.tensor(labels)

				ensemble_batch = torch.stack(emsemble_batches)

				state_object = {}

				for key, value in zip([example_numbers_key, labels_key, enemble_batch_key],[example_numbers, labels, ensemble_batch]):
					if key is not None:
						state_object[key] = value

				return state_object

		return collate_fn


	def roberta_single_sentence(*args, **kwargs):

		from transformers import RobertaTokenizer, AutoTokenizer, BartTokenizer

		tokenizer_types = {
			"auto":AutoTokenizer,
			"bart":BartTokenizer,
			"roberta_fast":RobertaTokenizer
		}

		tokenizer_type = tokenizer_types[kwargs.pop("tokenizer_type")]

		tokenizer_source = kwargs.pop("tokenizer_source")

		tokenizer = tokenizer_type.from_pretrained(tokenizer_source)		

		example_numbers_key = kwargs.pop("example_numbers_key", None)
		labels_key = kwargs.pop("labels_key", None)
		bert_batch_key = kwargs.pop("bert_batch_key")

		def collate_fn(data):

			with torch.no_grad():

				example_numbers, labels, sents = zip(*data)

				labels = torch.tensor(labels)

				bert_batch = tokenizer(sents, return_tensors='pt', padding=True)

				state_object = {}

				for key, value in zip([example_numbers_key, labels_key, bert_batch_key],[example_numbers, labels, bert_batch]):
					if key is not None:
						state_object[key] = value

				return state_object

		return collate_fn

	
	def stack_sentences(*args, **kwargs):

		example_numbers_key = kwargs.pop("example_numbers_key", None)
		labels_key = kwargs.pop("labels_key", None)
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

				state_object = {}

				for key, value in zip([example_numbers_key, labels_key, word_indexes_key, sentence_lengths_key],[example_numbers, labels, word_indexes, sentences_lenghts]):
					if key is not None:
						state_object[key] = value

				return state_object				

		return collate_fn	


	COLLATE_FN_DICT = {
		"allennlp_snli":allennlp_snli,
		"ensemble":ensemble,
		"ensemble_values":ensemble_values,
		"roberta_single_sentence":roberta_single_sentence,
		"stack_sentences":stack_sentences
	}
	

	def get_fn(*args, **kwargs):

		collate_fn_type = kwargs.pop('collate_fn_type')
		collate_fn = CollateFunctions.COLLATE_FN_DICT[collate_fn_type](**kwargs)

		return collate_fn

	
