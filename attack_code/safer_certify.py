# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.7.1       #
#       Created on: 2021-06-12      #
# --------------------------------- #


import copy
import math
import os
import pathlib
import pickle
from random import sample
import sys
import time

import numpy as np
from nltk.tokenize import word_tokenize
import torch

from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloaders.dataset_factory import DatasetFactory



class SaferCertify:

	def __init__(self, *args, **kwargs):

		self.synonyms = kwargs.pop("synonyms")
		self.B = kwargs.pop("B")
		self.word2idx = kwargs.pop("word2idx")
		self.vecs = kwargs.pop("vecs")
		self.examples = kwargs.pop("examples")

		self.num_classes = kwargs.pop("num_classes")
		self.P_size = kwargs.pop("pertubation_set_size")
		self.attack_budget = kwargs.pop("attack_budget")

		self.number_samples = kwargs.pop("number_samples", 5000)
		self.significance = kwargs.pop("significance", 0.01)

		self.estimate_error = 2*math.sqrt( ( math.log(1/self.significance) + math.log(self.num_classes) )  / (2*self.number_samples) )

		self.model =kwargs.pop("model")
		self.dataset_config =kwargs.pop("dataset_config")
		self.dataloader_params =kwargs.pop("dataloader_params")
		self.probs_key = kwargs.pop("prob_key")

		self.github_condition = kwargs.pop("github_condition", False)

		self.p_set_cache_file = kwargs.pop("p_set_cache_file", None)

		if (self.p_set_cache_file is not None) and (os.path.exists(self.p_set_cache_file)):
			with open(self.p_set_cache_file, "rb") as f:
				self.p_set_cache = pickle.load(f)

		else:
			self.p_set_cache = {}

		#so we know our cerifiable accuracy
		self.num_examples = len(self.examples)


	def RHS_condition(self, p_X):

		if self.github_condition:
			return p_X + 0.5 + self.estimate_error
		else:
			return 2*p_X + self.estimate_error


	def parse_examples(self):

		self.word_p_sets = {}
		self.example_vocab = {}

		for num, details in self.examples.items():

			sentence_tokens = [word_tokenize(s) for s in details["sentences"]]
			details["tokenized"] = sentence_tokens

			for s in sentence_tokens:
				for w in s:
					self.word_p_sets[w] = {}
					self.example_vocab[w] = None
					
					#need to find the p set for these as well to find q_x
					if w.lower() in self.synonyms:						
						for w_2 in self.synonyms[w.lower()]:
							self.word_p_sets[w_2] = {}

	#K = self.P_size
	#vectors already normalized
	def find_top_K_neighbours(self, word, bset):

		#if word is OOV we can't calulate q_x
		if word not in self.word2idx:
			return []

		word_vec = self.vecs[self.word2idx[word]]

		#can only compare negihbouts that are not OOV
		b_vecs = {}
		for w in bset:
			if w in self.word2idx:
				b_vecs[w] = self.vecs[self.word2idx[w]]


		vecs = np.stack([v for _, v in b_vecs.items()])

		dot_p = np.matmul(vecs, word_vec)

		best = np.argpartition(-dot_p, self.P_size)[:self.P_size]

		p_words = [w for w in b_vecs]

		p_set = [p_words[i] for i in best]

		return p_set
		

	#finds the pertubation set of each word in the example set
	def find_P_sets(self):

		print()
		print("Finding permutation sets...")
		print()

		dispaly_count = 50

		for w_i, word in enumerate(self.word_p_sets):

			#if this is in the cache don't do it again
			if word in self.p_set_cache:
				self.word_p_sets[word] = self.p_set_cache[word]

			else:

				#if the word has no b_set then it has no pertubation candidates, q_x=1
				if word not in self.B["w2set"]:
					self.word_p_sets[word] = {"p_set":[], "q_x":1}

				else:

					bs_i = self.B["w2set"][word]
					bset = self.B["sets"][bs_i]

					#if the b_set is smaller than the maximum pertubation set size choose them all
					#all synonyms have the same set so q_x = 1
					if len(bset) <= self.P_size:

						self.word_p_sets[word] = {"p_set":bset, "q_x":1}

					else:

						self.word_p_sets[word] = {"p_set":self.find_top_K_neighbours(word, bset)}

				#cache p_set for later
				self.p_set_cache[word] = self.word_p_sets[word]

			if ((w_i) % dispaly_count) == 0:
				print(f"\r {w_i}/{len(self.word_p_sets)}    ", end='\r')

		print(f"\r {len(self.word_p_sets)}/{len(self.word_p_sets)}    ")
		print()
		print()


	#only need q_xi for words in vocabulary of example
	def calc_q_xi(self):

		for word in self.example_vocab:

			details = self.word_p_sets[word]
			
			#if we need to calculate q_x for this word
			if "q_x" not in details:

				p_set = details["p_set"]

				syn_set = self.synonyms[word.lower()]

				p_set = set(p_set)

				min_overlap = float('inf')

				#see formula for q_x
				for syn in syn_set:

					syn_p_set = set(self.word_p_sets[syn]["p_set"])
					overlap = len(p_set.intersection(syn_p_set))

					if overlap < min_overlap:
						min_overlap = overlap

				details["q_x"] = min_overlap/len(p_set)

			#update cache
			self.p_set_cache[word] = details

		#clean up things we don't need anymore
		self.example_vocab ={word:self.word_p_sets[word] for word in self.example_vocab}
		self.word_p_sets = None


	def save_p_set_cache(self):

		output_folder = os.path.dirname(self.p_set_cache_file)

		pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

		with open(self.p_set_cache_file, "wb") as f:
			pickle.dump(self.p_set_cache, f)

		self.p_set_cache = None


	def calc_q_X(self):

		for num, details in self.examples.items():

			q_xis = []

			for s_i, s in enumerate(details["tokenized"]):

				for w_i, w in enumerate(s):

					q_xis.append(self.example_vocab[w]["q_x"])

			#as per definition of qX
			q_xis = sorted(q_xis)[:min(self.attack_budget, len(q_xis))]

			details["q_X"] = 1 - np.prod(q_xis)			


	#if 2*q_X+err > 1 then the example is uncertifiable independant of model architecture
	#formula A-B-2q_X-err > 0 wheren 1>A,B>0
	def filter_uncertifiable_by_pX(self):

		uncertifiable = []

		for num, details in self.examples.items():

			qX = details["q_X"]

			if self.RHS_condition(qX) > 1:
				uncertifiable.append(num)

		print()
		print(f"{len(uncertifiable)} examples are uncertifiable regardless of architecture")
		print()

		for num in uncertifiable:
			self.examples.pop(num)

		self.never_certifiable = len(uncertifiable)


	#only do attacks on words that have 
	def find_attack_targets(self):

		for num, details in self.examples.items():

			targets = []

			for s_i, s in enumerate(details["tokenized"]):

				for w_i, w in enumerate(s):

					p_set = self.example_vocab[w]["p_set"]

					if len(p_set) > 0:

						targets.append((s_i, w_i))

			details["attack_targets"] = targets


	def generate_adversary(self, example_number, attack_list):

		tokenized_sentence = copy.deepcopy(self.examples[example_number]["tokenized"])		

		for target_pos, target_word in attack_list:

			s_i, w_i = target_pos

			tokenized_sentence[s_i][w_i] = target_word

		#minor issue with puncutation here
		sentences = [" ".join(sent) for sent in tokenized_sentence]
		sentences = "\t".join(sentences)

		true_label = self.examples[example_number]["true_label_text"]

		attack_example = f"{example_number}\t{true_label}\t{sentences}"

		return attack_example


	def classify_examples(self,corpus, true_label, qX):

		remaining_examples = len(corpus)

		prediction_hist ={label:0 for label in range(self.num_classes)}

		with torch.no_grad():

			self.dataset_config["corpus"] = corpus

			dataset = DatasetFactory.BUILD_DATASET(**self.dataset_config)
			if dataset.collate_fn is not None:
				self.dataloader_params["collate_fn"] = dataset.collate_fn
			dataloader = DataLoader(dataset, **self.dataloader_params)

			probs = []

			for batch in dataloader:

				state_object = self.model(batch)
				batch_probs = state_object[self.probs_key].cpu().numpy()

				preds = np.argmax(batch_probs,axis=1)

				for p in preds:
					prediction_hist[p] += 1

				remaining_examples -= len(preds)

				#early stopping check
				correct_pred = prediction_hist.get(true_label)

				counter_pred = max(list([value for label, value in prediction_hist.items() if label != true_label]))

				#certifiable early stopping check
				if (correct_pred - counter_pred - remaining_examples)/self.number_samples > self.RHS_condition(qX):
					return prediction_hist

				#uncertifiable early stopping check
				if (correct_pred - counter_pred + remaining_examples)/self.number_samples <= self.RHS_condition(qX):
					return prediction_hist


	def certify_examples(self):

		uncertifiable = []

		print()
		print("Certifing examples...")
		print()

		print(f"\r Certifing examples {0}/{len(self.examples)}    ", end='\r')

		for n_i, (num, details) in enumerate(self.examples.items()):

			attack_corpus = []

			attack_targets = details["attack_targets"]
			tokenized_sentence = details["tokenized"]

			for _ in range(self.number_samples):

				current_targets = sample(attack_targets, min(self.attack_budget, len(attack_targets)))
				clean_words = [tokenized_sentence[s_i][w_i] for s_i, w_i in current_targets]
				attack_words = [sample(self.example_vocab[word]["p_set"], 1)[0] for word in clean_words]

				attack_list = [(target, word) for target, word in zip(current_targets, attack_words)]

				attack_example = self.generate_adversary(num, attack_list)

				attack_corpus.append(attack_example)

			true_label = details["true_label"]

			qX = details["q_X"]

			prediction_hist = self.classify_examples(attack_corpus, true_label, qX)

			correct_pred = prediction_hist.pop(true_label)

			counter_pred = max(list(prediction_hist.values()))

			#estimated g differnted in eq 2
			g = (correct_pred - counter_pred)/self.number_samples			

			#if the difference is not > 2qX + error the example is not 99% chance of certified
			if g <= self.RHS_condition(qX):
				uncertifiable.append(num)			

			print(f"\r Certifing examples {n_i+1}/{len(self.examples)}    ", end='\r')

		print(f"\r Certifing examples {len(self.examples)}/{len(self.examples)}    ")
		print()
		print()

		for num in uncertifiable:
			self.examples.pop(num)


	def generate_report(self,total_time):

		report = f"Total Certification Time {total_time}\n"

		report += f"Number Examples {self.num_examples}\n"

		cert_acc = len(self.examples)/self.num_examples

		report += f"Full Certified Accuracy {cert_acc}\n"

		cert_acc = len(self.examples)/(self.num_examples - self.never_certifiable)

		report += f"Certified Accuracy {cert_acc}\n"

		report += f"Example that are never certifiable {self.never_certifiable}\n"

		report += f"Attack Budget {self.attack_budget}\n"

		report += f"L = {self.P_size} N = {self.number_samples} delta = {self.significance}\n"\

		return report


	def cert(self):

		self.parse_examples()

		self.find_P_sets()

		self.calc_q_xi()

		self.save_p_set_cache()

		self.calc_q_X()

		self.filter_uncertifiable_by_pX()

		self.find_attack_targets()

		start_time = time.time()

		self.certify_examples()	

		total_time = time.time() - start_time		

		report = self.generate_report(total_time)

		return report


if __name__ == '__main__':

		
	result = safer.cert()

