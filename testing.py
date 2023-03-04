# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.7.1       #
#       Created on: 2021-06-12      #
# --------------------------------- #


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


model_trainer_config = {
	"number_of_classes":3,
	"max_rounds":10,
	"tenacity":10,
	"batch_loss_key":"loss_value",	
	"model_output_keys":
	{
		"batch_loss":"loss_value",
		"true_labels":"true_labels",
		"class_predictions":"class_predictions"
	},
	"dataset_config":
	{	
		"dataset_type": "snli_dataset",
		"corpus_pkl_file": "../corpus_data/snli_1.0_word2sense/train.pkl",	
		"indexes_key":"example_indexes", 
		"labels_key":"true_labels", 
		"word_indexes_key":"word_indexes",
		"sentence_lengths_key":"sentence_lengths"
	},
	"dataloader_params":
	{
		"batch_size":128,
		"shuffle":True,
		"num_workers":8
	},
	"display_config":
	{
		"max_train_buffer_length":250,		
	},
	"inference_evaluator_config":
	{
		"dataset_config":
		{	
			"dataset_type": "snli_dataset",
			"corpus_pkl_file": "../corpus_data/snli_1.0_word2sense/dev.pkl",

			"indexes_key":"example_indexes", 		
			"labels_key":"true_labels", 
			"word_indexes_key":"word_indexes",
			"sentence_lengths_key":"sentence_lengths"
		},
		"dataloader_params":
		{
			"batch_size":128,
			"shuffle":True,
			"num_workers":8
		},
	},
	"optimizer_config":
	{
		"optimizer_params":
		{
			"optimizer_type":"adam",
			"lr":0.001
		},
		"scheduler_params":
		{
			"scheduler_type":"steplr",
			"step_size":1,
			"gamma":0.5,
			"verbose":True
		}
	}
}

singleclass_model_config = {
	"module_type":"singleclass_model",
	"number_of_classes":3,
	"prediction_logits_key":"output_logits",
	"true_labels_key":"true_labels",
	"loss_values_key":"loss_value",
	"class_predictions_key":"class_predictions",
	"loss_fn":
	{
		"loss_fn_type":"cross_entropy"
	},
	"prediction_fn":
	{
		"prediction_fn_type":"max"
	},
	"module_config":
	{
		"module_type":"granular_model",
		"input_dimension":1,
		"submodule_configs" : 
		[
			{
				"module_type":"word2vec",
				"vector_pkl_file": "../corpus_data/snli_1.0_word2sense/vecs.pkl",
				"index_key": "word_indexes",
				"output_key": "batch_tensor"
			},
			{		
				"module_type":"ffn",
				"dropout_probability": 0.0,	
				"output_dimensions":[1024],
				"bias":False,
				"vector_batch_key": "batch_tensor",
				"output_key": "batch_tensor"
			},			
			{
				"module_type":"flat2pad",
				"padding_value": 0.0,
				"flat_tensor_key":"batch_tensor",
				"output_key":"batch_tensor",
				"sentence_length_key":"sentence_lengths"
			},
			{
				"module_type":"lstm",
				"num_layers":1,
				"padded_sentences_key":"batch_tensor",
				"output_key":"batch_tensor",
				"sentence_length_key":"sentence_lengths"
			},
			{		
				"module_type":"sentence_aggregation",
				"aggregation_type":"meanpooling",				
				"padded_sentences_key":"batch_tensor",
				"output_key":"batch_tensor", 
				"sentence_length_key":"sentence_lengths"
			},
			{		
				"module_type":"split_sentence_pairs",
				"flat_tensor_key":"batch_tensor",
				"output1_key":"batch_tensor1",
				"output2_key":"batch_tensor2"
			},
			{		
				"module_type":"sentence_pair_combination",
				"combination_type_list":["concatenation","absolute_difference","pairwise_multiplication"],
				"vector_batch1_key":"batch_tensor1",
				"vector_batch2_key":"batch_tensor2", 
				"output_key":"batch_tensor"
			},
			{		
				"module_type":"ffn",
				"dropout_probability":0.1,
				"output_dimensions":[1024,3],		
				"vector_batch_key": "batch_tensor",
				"output_key": "output_logits"		
			}
		]
	}
}

singleclass_model_config2 = {
	"module_type":"singleclass_model",
	"number_of_classes":3,
	"prediction_logits_key":"output_logits",
	"true_labels_key":"true_labels",
	"loss_values_key":"loss_value",
	"class_predictions_key":"class_predictions",
	"loss_fn":
	{
		"loss_fn_type":"cross_entropy"
	},
	"prediction_fn":
	{
		"prediction_fn_type":"max"
	},
	"module_config":
	{
		"module_type":"granular_model",
		"input_dimension":1,
		"submodule_configs" : 
		[
			{
				"module_type":"word2vec",
				"vector_pkl_file": "../corpus_data/snli_1.0_word2sense/vecs.pkl",
				"index_key": "word_indexes",
				"output_key": "batch_tensor"
			},
			{		
				"module_type":"ffn",
				"dropout_probability": 0.0,	
				"output_dimensions":[1024,1024,1024],
				"bias":False,
				"vector_batch_key": "batch_tensor",
				"output_key": "batch_tensor"
			},			
			{
				"module_type":"flat2pad",
				"padding_value": 0.0,
				"flat_tensor_key":"batch_tensor",
				"output_key":"batch_tensor",
				"sentence_length_key":"sentence_lengths"
			},
			{		
				"module_type":"sentence_aggregation",
				"aggregation_type":"meanpooling",				
				"padded_sentences_key":"batch_tensor",
				"output_key":"batch_tensor", 
				"sentence_length_key":"sentence_lengths"
			},
			{		
				"module_type":"split_sentence_pairs",
				"flat_tensor_key":"batch_tensor",
				"output1_key":"batch_tensor1",
				"output2_key":"batch_tensor2"
			},
			{		
				"module_type":"sentence_pair_combination",
				"combination_type_list":["concatenation","absolute_difference","pairwise_multiplication"],
				"vector_batch1_key":"batch_tensor1",
				"vector_batch2_key":"batch_tensor2", 
				"output_key":"batch_tensor"
			},
			{		
				"module_type":"ffn",
				"dropout_probability":0.1,
				"output_dimensions":[2048,1024,512,3],	
				"vector_batch_key": "batch_tensor",
				"output_key": "output_logits"		
			}
		]
	}
}

singleclass_model_config3 = {
	"module_type":"singleclass_model",
	"number_of_classes":3,
	"prediction_logits_key":"output_logits",
	"true_labels_key":"true_labels",
	"loss_values_key":"loss_value",
	"class_predictions_key":"class_predictions",
	"loss_fn":
	{
		"loss_fn_type":"cross_entropy"
	},
	"prediction_fn":
	{
		"prediction_fn_type":"max"
	},
	"module_config":
	{
		"module_type":"granular_model",
		"input_dimension":1,
		"submodule_configs" : 
		[
			{
				"module_type":"word2vec",
				"vector_pkl_file": "../corpus_data/snli_1.0_word2sense/vecs.pkl",
				"index_key": "word_indexes",
				"output_key": "batch_tensor"
			},
			{		
				"module_type":"ffn",
				"dropout_probability": 0.0,	
				"output_dimensions":[128],
				"bias":False,
				"vector_batch_key": "batch_tensor",
				"output_key": "batch_tensor"
			},			
			{
				"module_type":"flat2pad",
				"padding_value": 0.0,
				"flat_tensor_key":"batch_tensor",
				"output_key":"batch_tensor",
				"sentence_length_key":"sentence_lengths"
			},
			{		
				"module_type":"sentence_aggregation",
				"aggregation_type":"meanpooling",				
				"padded_sentences_key":"batch_tensor",
				"output_key":"batch_tensor", 
				"sentence_length_key":"sentence_lengths"
			},
			{		
				"module_type":"split_sentence_pairs",
				"flat_tensor_key":"batch_tensor",
				"output1_key":"batch_tensor1",
				"output2_key":"batch_tensor2"
			},
			{		
				"module_type":"sentence_pair_combination",
				"combination_type_list":["concatenation","absolute_difference","pairwise_multiplication"],
				"vector_batch1_key":"batch_tensor1",
				"vector_batch2_key":"batch_tensor2", 
				"output_key":"batch_tensor"
			},
			{		
				"module_type":"ffn",
				"dropout_probability":0.1,
				"output_dimensions":[3],	
				"vector_batch_key": "batch_tensor",
				"output_key": "output_logits"		
			}
		]
	}
}

def build_model(mode):
	
	if mode == 1:
		model = ModuleFactory.BUILD_MODULE(**singleclass_model_config)
	elif mode == 2:
		model = ModuleFactory.BUILD_MODULE(**singleclass_model_config2)
	elif mode == 3:
		model = ModuleFactory.BUILD_MODULE(**singleclass_model_config3)

	model.to_device()

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])	

	print()
	print(model)
	print(f"Number of Paramerter: {params}")
	print()
	print()

	return model


def test_model_building(model):

	with torch.no_grad():

		print(model)

		state_object = {"word_indexes":torch.tensor([1,2,3,4,5,6,7,8,9,10]), "sentence_lengths":[1,2,3,4], "true_labels":torch.tensor([0,1])}

		print(model(state_object))


def build_IE():

	builder = InferenceEvaluatorBuilder()
	builder.configure(**IE_config)
	inference_evaluator = builder.build()

	return inference_evaluator

def do_epoch(model, inference_evaluator):

	results = inference_evaluator(model)

	return results


def test_epoch(model, inference_evaluator):

	results = do_epoch(model, inference_evaluator)

	return results


def check_avg_time(model, inference_evaluator, iterations):

	times = []

	for _ in range(iterations):		

		results = do_epoch(model, inference_evaluator)

		times.append(results["inference_time"])

	mean = sum(times)/len(times)
	std = math.sqrt(sum([(t - mean)**2 for t in times])/len(times))

	print(mean, std)


def training_loop(model, model_trainer):

	model_trainer(model)


def model_trainer_building_test():

	model_trainer_builder = ModelTrainerBuilder()
	model_trainer_builder.configure(**model_trainer_config)
	model_trainer = model_trainer_builder.build()

	return model_trainer


def main():

	working_folder="../experiments/testing/"

	pathlib.Path(working_folder).mkdir(parents=True, exist_ok=True)

	config_folder = os.path.join(working_folder, "configs/")

	os.mkdir(config_folder)

	model_trainer_config["checkpoint_folder"] = working_folder
	model_trainer_config["display_config"]["log_file"] = os.path.join(working_folder, "training_log.txt")
	model_trainer_config["display_config"]["summary_file"] = os.path.join(working_folder, "summary_test.txt")

	iterations = 10
	num_epochs = 20
	RANDOM_SEED = 1234

	torch.manual_seed(RANDOM_SEED)

	model = build_model(3)

	#inference_evaluator = build_IE()	

	#test_model_building(model)

	#print(test_epoch(model, inference_evaluator))

	#check_avg_time(model, inference_evaluator, iterations)

	model_trainer = model_trainer_building_test()

	training_loop(model, model_trainer)


if __name__ == '__main__':
	main()