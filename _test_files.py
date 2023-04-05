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




def main():

	model_config = {
		"module_type":"hf_model",
		"class":"auto_seq_class",
		"address":"cross-encoder/nli-deberta-v3-xsmall",
		"batch_key":"bert_batch",
		"output_key":"output_logits",
		"output_dimension":3
	}

	model_config = {
		"module_type":"hf_model",
		"class":"roberta_adapter",
		"address":"roberta-base",
		"adapter":"AdapterHub/roberta-base-pf-snli",
		"batch_key":"bert_batch",
		"output_key":"output_logits",
		"output_dimension":3
	}

	with torch.no_grad():


		love = "I love you."
		hate = "I hate you."
		dog = "the dog is large"

		noise_dist = torch.distributions.normal.Normal(torch.tensor([0.0], device="cuda"),torch.tensor([0.1], device="cuda"))
		noise = noise_dist.sample([32,3])

		print(noise)
		print(noise.size())

		values = torch.full((32,3,1), 0.3, device="cuda")

		print(values)
		print(values.size())

		noisy_values = noise + values

		torch.nn.ReLU(noisy_values)

		noisy_values_sum = torch.sum(noisy_values, dim=1)

		print(noisy_values_sum)
		print(noisy_values_sum.size())

		noisy_values_normalized = noisy_values/noisy_values_sum.unsqueeze(1)

		print(noisy_values_normalized)
		print(noisy_values_normalized.size())

		print()

		print(dir(noise_dist))
		print(noise_dist.mean)
		print(noise_dist.stddev)



	



if __name__ == '__main__':
	main()