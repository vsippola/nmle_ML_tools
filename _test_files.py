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

model_trainer_config_test = {
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
		"dataset_type": "w2v_sentence_pair",
		"corpus_pkl_file": "../parsed_data/snli_1.0/snli_1.0_train.tsv",	
		"word2idx_file":"../",
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


def test_epoch(model, inference_evaluator):

	results = do_epoch(model, inference_evaluator)

	return results




def training_loop(model, model_trainer):

	model_trainer(model)


def model_trainer_building_test(model_trainer_config):

	model_trainer_builder = ModelTrainerBuilder()
	model_trainer_builder.configure(**model_trainer_config)
	model_trainer = model_trainer_builder.build()

	return model_trainer


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

	test_folder = "../experiments/snli_roberta/"
	fname_pre = "Checkpoint_"
	fname_ext = ".tar"
	test_files = [0,1,2,3,4,5,6,7,8,10,12,14,15,16,18,19,20,21,32,33,35,38]

	for fname in test_files:

		checkpoint_fname = os.path.join(test_folder, f"{fname_pre}{fname}{fname_ext}")

		testing_loop(inference_config, model, checkpoint_fname)


def test_HF():

	from transformers import RobertaAdapterModel

	config = {
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
			"module_type":"hf_roberta_adapter",
			"roberta_base":"roberta-base",
			"adapter_name":"AdapterHub/roberta-base-pf-snli",
			"bert_batch_key":"bert_batch",
			"output_key":"output_logits"
		}
	}



	#model = ModuleFactory.BUILD_MODULE(**config)
	#model.to_device()

	with torch.no_grad():
		from transformers import RobertaTokenizerFast
		tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
		sents = [" I love you. I love you."," I love you. I hate you.", "The dog exists and is not clean. The dog smells funny."]
		bert_input = tokenizer(sents, return_tensors='pt', padding=True)

		print()
		print(bert_input)
		print()

		indexes = bert_input['input_ids']
		print()
		print(indexes)
		print()
		print(indexes.size())

		sys.exit()

		state_object = {"bert_batch":bert_input, "true_labels":torch.tensor([0,2,1])}
		state_object = model(state_object)
		print(state_object)


def run_no_train_test(model):

	inference_evaluator_builder = InferenceEvaluatorBuilder()
	inference_evaluator_builder.configure(**inference_config)
	inference_evaluator = inference_evaluator_builder.build()

	output = inference_evaluator(model)
	print()
	print(output)


def test_elmo():
	from allennlp_models.pretrained import load_predictor 

	predictor = load_predictor("pair-classification-decomposable-attention-elmo")

	premise = "A man in a black shirt overlooking bike maintenance."
	hypothesis = "A man destroys a bike."
	json_instance = [{"premise":premise, "hypothesis":hypothesis}, {"premise":premise, "hypothesis":hypothesis}]
	preds = predictor.predict_batch_json(json_instance)

	print(preds)


	

def main():

	working_folder="../experiments/snli_bilstm_4"
	config_folder = "../configs/snli_bilstm/"	

	model_config_fname = os.path.join(config_folder, "model.json")

	with open(model_config_fname, 'r') as f:
		model_config = json.load(f)

	model_trainer_config_fname = os.path.join(config_folder, "training.json")

	with open(model_trainer_config_fname, 'r') as f:
		model_trainer_config = json.load(f)


	pathlib.Path(working_folder).mkdir(parents=True, exist_ok=True)

	config_folder = os.path.join(working_folder, "configs/")

	pathlib.Path(config_folder).mkdir(parents=True, exist_ok=True)	

	for config, fname in zip([model_config, model_trainer_config],["model.json", "training.json"]):

		output_filename = os.path.join(config_folder, fname)
		json_object = json.dumps(config, indent=4)
		with open(output_filename, 'w') as f:
			f.write(json_object)

	model_trainer_config["checkpoint_folder"] = working_folder
	model_trainer_config["display_config"]["log_file"] = os.path.join(working_folder, "training_log.txt")
	model_trainer_config["display_config"]["summary_file"] = os.path.join(working_folder, "summary_test.txt")

	RANDOM_SEED = 1234

	torch.manual_seed(RANDOM_SEED)

	model = build_model(model_config)

	model_trainer = model_trainer_building_test(model_trainer_config)

	training_loop(model, model_trainer)

	#testing_loop

	#run_testing(model)

	#run_no_train_test(model)
	
	#test_HF()

	#test_elmo()

	



if __name__ == '__main__':
	main()