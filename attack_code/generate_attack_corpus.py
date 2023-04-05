# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.7.1       #
#       Created on: 2021-06-12      #
# --------------------------------- #


def load_coprus(corpus_text_file, corpus_indexes, label2int=None):

	text_corpus = {}

	with open(corpus_text_file, "r") as f:

		for corpus_line in f:

			corpus_line_tokens = corpus_line.strip("\n").split("\t")

			example_num = int(corpus_line_tokens[corpus_indexes["example_num"]])
			true_label_text = corpus_line_tokens[corpus_indexes["true_label"]]			
			true_label = label2int[true_label_text]
			sentences = [corpus_line_tokens[idx] for idx in corpus_indexes["sentences"]]

			example = {
				"number":example_num,
				"true_label_text":true_label_text,
				"true_label":true_label,
				"sentences":sentences
			}

			text_corpus[example_num] = example

	return text_corpus


def load_prediction_details(prediction_details_file, prediction_details_indexes):

	prediction_details = {}

	with open(prediction_details_file, "r") as f:

		for prediction_details_line in f:

			prediction_details_line_tokens = prediction_details_line.strip("\n").split("\t")

			example_num = int(prediction_details_line_tokens[prediction_details_indexes["example_num"]])
			pred_label = int(prediction_details_line_tokens[prediction_details_indexes["pred_label"]])
			probs = prediction_details_line_tokens[prediction_details_indexes["probs"]]
			probs = [float(p) for p in probs.split(",")]

			details = {
				"pred_label":pred_label,
				"probs":probs
			}

			prediction_details[example_num] = details

	return prediction_details


def filter_correct(attack_corpus):

	new_attack_corpus = {}

	for num, details in attack_corpus.items():

		true_label = details["true_label"]
		pred_label = details["pred_label"]

		if true_label == pred_label:
			new_attack_corpus[num] = details

	return new_attack_corpus


def get_attack_examples(attack_corpus, attack_strength, number):

	start_idx = attack_strength*number
	end_idx = start_idx + number

	if start_idx > len(attack_corpus):
		return []

	if end_idx > len(attack_corpus):
		end_idx = len(attack_corpus)

	return dict(attack_corpus[start_idx:end_idx])


def generate_attack_corpus(*args, **kwargs):

	number_of_examples = kwargs.pop("number_of_examples")

	attack_strength = kwargs.pop("attack_strength")

	corpus_text_file = kwargs.pop("corpus_text_file")
	corpus_indexes = kwargs.pop("corpus_indexes")
	label2int = kwargs.pop("label2int", None)

	prediction_details_file = kwargs.pop("prediction_details_file")
	prediction_details_indexes = kwargs.pop("prediction_details_indexes")

	use_correct_preds = kwargs.pop("use_correct_preds", True)


	text_corpus = load_coprus(corpus_text_file, corpus_indexes, label2int)

	predictions_details = load_prediction_details(prediction_details_file, prediction_details_indexes)

	attack_corpus = {example_num: text_corpus[example_num] | predictions_details[example_num] for example_num in text_corpus }

	if use_correct_preds:
		attack_corpus = filter_correct(attack_corpus)

	for num, details in attack_corpus.items():

		pred_label = details["pred_label"]
		attack_corpus[num]["p_pred"] = details["probs"][pred_label]

	#sort by prediction score (idea higher score = harder to attack?)
	attack_corpus = sorted(attack_corpus.items(), key=lambda x:x[1]["p_pred"], reverse=True)
	
	attack_corpus = get_attack_examples(attack_corpus, attack_strength, number_of_examples)

	return attack_corpus

