# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.7.1       #
#       Created on: 2021-06-12      #
# --------------------------------- #

import json
import math
import os
import pathlib
import pickle
import time

import numpy as np

import torch
from torch.utils.data import DataLoader



from models.builders.module_factory import ModuleFactory

from training.model_trainer_builder import ModelTrainerBuilder

from training.inference_evaluator_builder import InferenceEvaluatorBuilder

from training.model_output_metrics import ModelPredictionMetrics
from training.display_training_results import DisplayTrainingResults
from training.model_trainer import ModelTrainer
from dataloaders.dataset_factory import DatasetFactory

inference_config = {
	"time_key":"test_time",
	"dataset_config":
	{			
		"dataset_type": "sentence_list",
		"corpus_file": "../parsed_data/snli_1.0/snli_1.0_test.tsv",	

		"example_number_index":0,

		"label_transform_params":
		{
			"label_transform_type":"dict",
			"label_index":1,
			"label_dict":
			{
				"entailment":0,
				"neutral":1,
				"contradiction":2
			}
		},

		"sentence_transform_params":
		{
			"sentence_transform_type":"w2v",
			"word2idx_file":"../word2vec_pickles/fasttext_crawl-300d-2M/snli_1.0/vocab.pkl",
			"sentence_index":[2, 3],
			"combine_before_transform":False
		},

		"collate_fn_params":
		{
			"collate_fn_type":"stack_sentences",
			"example_numbers_key":"example_numbers", 
			"labels_key":"true_labels", 
			"word_indexes_key":"word_indexes",
			"sentence_lengths_key":"sentence_lengths"
		}		
	},
	"dataloader_params":
	{
		"batch_size":128,
		"shuffle":True,
		"num_workers":8
	},
	"metric_tracker_params":
	{
		"number_of_classes":3,
		"model_output_keys":
		{
			"batch_loss":"loss_value",
			"true_labels":"true_labels",
			"class_predictions":"class_predictions"
		},
		"metric_output_keys":{
			"loss":"test_loss",
			"accuracy":"test_acc"
		}
	}	
}



def build_model(model_config):

	model = ModuleFactory.BUILD_MODULE(**model_config)

	model.to_device()

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])	

	print()
	#print(model)
	print(f"Number of Paramerter: {params}")
	print()
	print()

	return model


def testing_loop(inference_config, model, checkpoint_fname):

	inference_evaluator_builder = InferenceEvaluatorBuilder()
	inference_evaluator_builder.configure(**inference_config)
	inference_evaluator = inference_evaluator_builder.build()

	checkpoint = torch.load(checkpoint_fname)
	model.load_from_dict(**checkpoint["model_params"])

	output = inference_evaluator(model)
	print()
	print(checkpoint_fname)
	print(output)


def run_testing(model):

	test_folder = "../experiments/snli_bilstm_4/"
	fname_pre = "Checkpoint_"
	fname_ext = ".tar"
	test_files = [0,1,2,3,4,5,6,7,8,9,10,12,13,15,18,20,21,22,23,34,35,56,59,70]

	for fname in test_files:

		checkpoint_fname = os.path.join(test_folder, f"{fname_pre}{fname}{fname_ext}")

		testing_loop(inference_config, model, checkpoint_fname)


def run_no_train_test(inference_config, model):

	inference_evaluator_builder = InferenceEvaluatorBuilder()
	inference_evaluator_builder.configure(**inference_config)
	inference_evaluator = inference_evaluator_builder.build()

	output = inference_evaluator(model)
	print()
	print(output)


def test_ensemble(testing_config, model):

	dataset = DatasetFactory.BUILD_DATASET(**testing_config)

	dataloader_params = {
		"batch_size":32,
		"shuffle":True,
		"num_workers":7,
		"collate_fn":dataset.collate_fn
	}

	dataloader = DataLoader(dataset, **dataloader_params)

	model.eval()

	with torch.no_grad():

		stime = time.time()
		for batch in dataloader:
			model(batch)

		print( (time.time()-stime)/60.0)


def main():

	from transformers import RobertaTokenizerFast	

	tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")	

	print(dir(tokenizer))

	print(tokenizer.sep_token, tokenizer(tokenizer.sep_token))	
	print(tokenizer.cls_token, tokenizer(tokenizer.cls_token))


	love = "I love you."
	hate = "I hate you."
	dog = "the dog is large"

	sent = " ".join([love, dog])

	tokens = tokenizer(sent)

	print(tokens['input_ids'])

	sent = " </s> ".join([love, dog])
	sent = f"<s> {sent} </s>"

	tokens = tokenizer(sent)

	print(tokens['input_ids'])

	



if __name__ == '__main__':
	main()