# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.7.1       #
#       Created on: 2021-06-12      #
# --------------------------------- #

import copy
from math import exp
import time

from nltk.corpus import wordnet as wn
import numpy as np
import spacy
import torch

from torch.utils.data import DataLoader

from dataloaders.dataset_factory import DatasetFactory



class PWWSAttacker:

	#list form PWWS github
	SUPPORTED_POS_TAGS = [
		'CC',  # coordinating conjunction, like "and but neither versus whether yet so"
		# 'CD',   # Cardinal number, like "mid-1890 34 forty-two million dozen"
		# 'DT',   # Determiner, like all "an both those"
		# 'EX',   # Existential there, like "there"
		# 'FW',   # Foreign word
		# 'IN',   # Preposition or subordinating conjunction, like "among below into"
		'JJ',  # Adjective, like "second ill-mannered"
		'JJR',  # Adjective, comparative, like "colder"
		'JJS',  # Adjective, superlative, like "cheapest"
		# 'LS',   # List item marker, like "A B C D"
		# 'MD',   # Modal, like "can must shouldn't"
		'NN',  # Noun, singular or mass
		'NNS',  # Noun, plural
		'NNP',  # Proper noun, singular
		'NNPS',  # Proper noun, plural
		# 'PDT',  # Predeterminer, like "all both many"
		# 'POS',  # Possessive ending, like "'s"
		# 'PRP',  # Personal pronoun, like "hers herself ours they theirs"
		# 'PRP$',  # Possessive pronoun, like "hers his mine ours"
		'RB',  # Adverb
		'RBR',  # Adverb, comparative, like "lower heavier"
		'RBS',  # Adverb, superlative, like "best biggest"
		# 'RP',   # Particle, like "board about across around"
		# 'SYM',  # Symbol
		# 'TO',   # to
		# 'UH',   # Interjection, like "wow goody"
		'VB',  # Verb, base form
		'VBD',  # Verb, past tense
		'VBG',  # Verb, gerund or present participle
		'VBN',  # Verb, past participle
		'VBP',  # Verb, non-3rd person singular present
		'VBZ',  # Verb, 3rd person singular present
		# 'WDT',  # Wh-determiner, like "that what whatever which whichever"
		# 'WP',   # Wh-pronoun, like "that who"
		# 'WP$',  # Possessive wh-pronoun, like "whose"
		# 'WRB',  # Wh-adverb, like "however wherever whenever"
	]
	SUPPORTED_POS_TAGS = set(SUPPORTED_POS_TAGS)

	def __init__(self, *args, **kwargs):

		self.examples = kwargs.pop("examples")
		self.model = kwargs.pop("model")
		self.dataset_config = kwargs.pop("dataset_config")
		self.dataloader_params = kwargs.pop("dataloader_params")

		self.probs_key = kwargs.pop("prob_key")
		self.int2label = kwargs.pop("int2label")

		spacy.prefer_gpu()
		self.nlp = spacy.load("en_core_web_sm")

		self.attack_examples = {}
		self.safe_examples = {}




	def _found_attack(self, num, attack_list, attack_pred):

		#generate attack
		clean = self._generate_adversary(num, [])
		adversay = self._generate_adversary(num, attack_list)

		example = self.examples.pop(num)

		attacked_example = {
			"clean":clean,
			"adversay":adversay,
			"true_label":example["true_label"],
			"attack_pred":attack_pred,
			"number_substituions":len(attack_list),
			"sub_rate":len(attack_list)/len(example["parsed_sentences"]),
			"attack_list":attack_list,
			"p_prob":example["p_pred"]
		}

		self.attack_examples[num] = attacked_example


	def _mark_safe(self, num):

		clean = self._generate_adversary(num, [])

		example = self.examples.pop(num)

		safe_example = {
			"clean":clean,
			"true_label":example["true_label"],
		}

		self.safe_examples[num] = safe_example


	def _generate_probs(self, corpus):

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

				probs.append(batch_probs)
					
			probs = np.concatenate(probs)

			return probs


	#from PWWS github
	def _get_wordnet_pos(self, tag):
		'''Wordnet POS tag'''
		pos = tag.lower()
		if pos in ['r', 'n', 'v']:  # adv, noun, verb
			return pos
		elif pos == 'j':
			return 'a'  # adj


	#generates additional metadata to do attack
	def _parse_sentences(self):

		print()
		print("Parsing sentences...")
		print()

		display_count = 50

		for count, (num, details) in enumerate(self.examples.items()):

			sentences = details["sentences"]

			parsed_sentences = {}

			tokenized = []			

			attack_targets = {}

			for s_i, s in enumerate(sentences):				

				parsed_s = self.nlp(s)		

				sentences_tokenized = []

				for w_i, w in enumerate(parsed_s):

					parsed_word = {
						"w":w, 
						"wnet_pos":self._get_wordnet_pos(w.tag_)
					}

					parsed_sentences[(s_i, w_i)] = parsed_word

					if w.tag_ in PWWSAttacker.SUPPORTED_POS_TAGS:
						attack_targets[(s_i, w_i)] = None

					sentences_tokenized.append(w.text)

				tokenized.append(sentences_tokenized)

			self.examples[num]["parsed_sentences"] = parsed_sentences
			self.examples[num]["tokenized"] = tokenized
			self.examples[num]["attack_targets"] = attack_targets

			if (count+1) % display_count == 0:
				print(f"\rParsed {count +1}/{len(self.examples)}", end="\r")

		print(f"\rParsed {len(self.examples)}/{len(self.examples)}")

		print()


	#from PWWS github
	#I'm not sure the way they compare tags works? as the tag ffor the synonym is generated
	#without any context, not sure if we should be tagging it after replacing into the document?
	def _filter_synonyms(self, word, synonym):

		if (len(synonym.text.split()) > 2 or (  # the synonym produced is a phrase
			synonym.lemma == word.lemma) or (  # token and synonym are the same
			synonym.tag != word.tag) or (  # the pos of the token synonyms are different
			word.text.lower() == 'be')):  # token is be

			return False

		else:
			
			return True


	#gets the synonym set for one word
	def _get_single_synonyms_set(self, word):

		synsets = wn.synsets(word["w"].text, word["wnet_pos"])

		#get synonyms from synsets and remove duplicates
		synonyms = [self.nlp(syn.name().replace('_', ' '))[0] for synset in synsets for syn in synset.lemmas()]

		#filter based on pwws critera
		synonyms = [synonym for synonym in synonyms if self._filter_synonyms(word["w"], synonym)]

		return list(set([syn.text for syn in synonyms]))


	#gets the synonym set for all target words
	def _get_all_synonyms(self):

		print()
		print("Generating synonyms...")
		print()

		display_count = 10

		#to speed up synonym finding
		synonym_dict = {}

		for count, (num, details) in enumerate(self.examples.items()):

			target_synonyms = {}		

			for target in details["attack_targets"]:

				word = details["parsed_sentences"][target]

				word_key = (word["w"].lemma, word["w"].tag)

				if word_key in synonym_dict:

					target_synonyms[(target)] = synonym_dict[word_key]

				else:

					synonyms = self._get_single_synonyms_set(word)

					target_synonyms[(target)] = synonyms

					synonym_dict[word_key] = synonyms		


			for target, synonyms in target_synonyms.items():

				#if there are no synonyms remove the target
				if len(synonyms) == 0:

					details["attack_targets"].pop(target)					

				else:

					self.examples[num]["parsed_sentences"][target]["synonyms"] = synonyms

			if (count + 1) % display_count == 0:
				print(f"\rSynonyms found {count +1}/{len(self.examples)}", end="\r")

		print(f"\rSynonyms found {len(self.examples)}/{len(self.examples)}")

		print()


	def _calc_pwws_queries(self):

		pwws_queries = 0

		for num, details in self.examples.items():

			#one query for each word in the sentence for saliancy checks
			pwws_queries += len(details["parsed_sentences"])

			for target in details["attack_targets"]:

				#one query for each synonym
				pwws_queries += len(details["parsed_sentences"][target]["synonyms"])

		return pwws_queries


	def _generate_adversary(self, example_number, attack_list):

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


	def _find_best_synonyms(self):

		print()
		print("Finding best synonyms...")
		print()

		number_queries = 0

		attack_details = {}

		safe = []

		for num, details in self.examples.items():

			#if there are attack targets copy them
			if len(details["attack_targets"]) > 0:

				#make a copy of targets for each example
				targets = list(copy.deepcopy(details["attack_targets"]))

				#get the current target for example
				curr_target = targets.pop()

				synonyms = list(details["parsed_sentences"][curr_target].pop("synonyms"))

				curr_syn = synonyms.pop()

				true_label = details["true_label"]

				base_score = details["p_pred"]

				attack_details[num] = {
					"targets":targets,
					"curr_target":curr_target,
					"synonyms":synonyms,
					"curr_syn":curr_syn,
					"true_label":true_label,
					"base_score":base_score,
					"best_atack":-float("inf"),
					"best_syn":None
				}

			else:

				safe.append(num)

		for num in safe:
			self._mark_safe(num)

		count = 0
		display_count = 10

		while len(attack_details) > 0:

			attack_batch = []

			for num, details in attack_details.items():

				curr_target = details["curr_target"]
				curr_syn = details["curr_syn"]

				attack_example = self._generate_adversary(num, [(curr_target, curr_syn)])

				attack_batch.append(attack_example)

			number_queries += len(attack_batch)

			prob_list = self._generate_probs(attack_batch)

			attack_preds = np.argmax(prob_list, axis=1)

			successful = []
			done = []

			#evaluate the single word attacks
			for (num, details), probs, attack_pred in zip(attack_details.items(), prob_list, attack_preds):

				#if this attack succeded
				if attack_pred != details["true_label"]:

					curr_target = details["curr_target"]
					curr_syn = details["curr_syn"]

					attack_list = [(curr_target, curr_syn)]

					successful.append((num, attack_list, attack_pred))

				#otherwise calculate the attack effect
				else:

					prob = probs[details["true_label"]]

					#check if the new attack was the best
					prediciton_diff = details["base_score"] - prob
					if prediciton_diff > details["best_atack"]:
						details["best_syn"] = details["curr_syn"]
						details["best_atack"] = prediciton_diff

					#iterate through synonyms if we can
					synonyms = details["synonyms"]
					if len(synonyms) > 0:

						details["curr_syn"] = synonyms.pop()

					#otherwise we have attacked this target with all synonyms
					else:

						#update the target with it's best attack
						curr_target = details["curr_target"]
						self.examples[num]["parsed_sentences"][curr_target]["synonym"] = details["best_syn"]
						self.examples[num]["parsed_sentences"][curr_target]["attack_strength"] = details["best_atack"]

						#iterate through targets if we can:
						targets = details["targets"]

						if len(targets) > 0:

							curr_target = targets.pop()

							details["curr_target"] = curr_target
							details["synonyms"] = list(self.examples[num]["parsed_sentences"][curr_target].pop("synonyms"))
							details["curr_syn"] = details["synonyms"].pop()
							details["best_atack"] = -float("inf"),
							details["best_syn"] = None

						#otherwise this exmample is done
						else:

							done.append(num)			

			for num in done:
				attack_details.pop(num)

			for (num, attacklist, attack_pred) in successful:
				attack_details.pop(num)
				self._found_attack(num, attacklist, attack_pred)

			if count % display_count == 0:
				print(f"\rExamples remaining: {len(attack_details)}-----", end="\r")
			count += 1

		print(f"\rExamples remaining: {len(attack_details)}-----")
		print()

		return number_queries


	def _calculate_saliancies(self):

		print()
		print("Calculating Saliancies...")
		print()

		if len(self.examples) == 0:
			return 0

		#gererate attack details
		attack_details = []
		attack_batch = []
		for num, details in self.examples.items():

			base_score = details["p_pred"]
			true_label = details["true_label"]

			for target in details["attack_targets"]:

				attack = (num, target, base_score, true_label)
				attack_details.append(attack)

				attack_example = self._generate_adversary(num, [(target, "")])
				attack_batch.append(attack_example)

		number_queries = len(attack_details)

		prob_list = self._generate_probs(attack_batch)

		for (num, target, base_score, true_label), probs in zip(attack_details, prob_list):

			pred = probs[true_label]
			saliancy = base_score - pred

			self.examples[num]["parsed_sentences"][target]["saliancy"] = saliancy

		return number_queries


	def _generate_multiword_attacks(self):

		print()
		print("Generating multiword attacks...")
		print()

		if len(self.examples) == 0:
			return

		display_count = 50

		for count, (num, details) in enumerate(self.examples.items()):

			parsed_sentences = details[("parsed_sentences")]
			attack_targets = details["attack_targets"]

			saliancies = []
			for target in attack_targets:

				saliancy = parsed_sentences[target].pop("saliancy")
				saliancies.append(saliancy)

			saliancies = [exp(s) for s in saliancies]
			sum_s = sum(saliancies)
			saliancies = [s/sum_s for s in saliancies]

			attacks = []

			for target, s in zip(attack_targets, saliancies):

				synonym = parsed_sentences[target].pop("synonym")
				attack_strength = parsed_sentences[target].pop("attack_strength") * s

				attacks.append([(target, synonym), attack_strength])

			attacks = sorted(attacks, key=lambda t: t[1], reverse=True)

			attacks_list = [attack[0] for attack in attacks]

			self.examples[num]["attacks_list"] = attacks_list

			if (count + 1) % display_count == 0:
				print(f"\rSynonyms found {count +1}/{len(self.examples)}", end="\r")

		print(f"\r Multi-word attacks created {len(self.examples)}/{len(self.examples)}")

		print()


	def _peform_multiword_attacks(self):

		print()
		print("Performing multiword attacks...")
		print()

		if len(self.examples) == 0:
			return 0

		number_queries = 0

		attack_details = {}

		for num, details in self.examples.items():

			#make a copy of targets for each example
			full_attack_list = details.pop("attacks_list")

			true_label = details["true_label"]

			attack_details[num] = {
				"full_attack_list":full_attack_list,
				"true_label":true_label,
			}

		attack_size = 1

		count = 0
		display_count = 10

		while len(attack_details) > 0:	

			attack_batch = []
			attack_lists = []		

			for num, details in attack_details.items():

				attack_list = details["full_attack_list"][:attack_size]
				attack_lists.append(attack_list)

				attack_example = self._generate_adversary(num, attack_list)

				attack_batch.append(attack_example)

			number_queries += len(attack_batch)

			prob_list = self._generate_probs(attack_batch)

			attack_preds = np.argmax(prob_list, axis=1)

			successful = []
			
			for (num, details), pred, attack_list in zip(attack_details.items(), attack_preds, attack_lists):

				if pred != details["true_label"]:

					attack = (num, attack_list, pred)
					successful.append(attack)

			for num, attack_list, pred in successful:

				attack_details.pop(num)
				self._found_attack(num, attack_list, pred)

			attack_size += 1

			#remove exampes that are safe (no more attacks try)
			safe = []
			for num, details in attack_details.items():

				if attack_size > len(details["full_attack_list"]):

					safe.append(num)

			for num in safe:

				attack_details.pop(num)
				self._mark_safe(num)	

			if count % display_count == 0:
				print(f"\rExamples remaining: {len(attack_details)}-----", end="\r")
			count+=1

		print(f"\rExamples remaining: {len(attack_details)}-----")
		print()

		return number_queries

	def _generate_report(self, attack_time, total_queries_pwws, total_queries_mine):

		new_accuracy = len(self.safe_examples)/(len(self.safe_examples) + len(self.attack_examples))
		avg_queries_pwws = total_queries_pwws/(len(self.safe_examples) + len(self.attack_examples))
		avg_queries_mine = total_queries_mine/(len(self.safe_examples) + len(self.attack_examples))

		safe_report = ""

		for num, example in self.safe_examples.items():
			clean = example["clean"]
			safe_report += f"{clean}\n"

		attack_report = ""

		total_subs = 0.0
		total_subrate = 0.0
		total_prob = 0.0

		for num, example in self.attack_examples.items():

			clean = example["clean"]
			adversay = example["adversay"]
			true_label = self.int2label[example["true_label"]]
			attack_pred = self.int2label[example["attack_pred"]]			
			number_substituions = example["number_substituions"]
			sub_rate = example["sub_rate"]
			attack_list = example["attack_list"]
			p_prob = example["p_prob"]

			report = f"Original examples: {clean}\nAdversarial Example: {adversay}\n"+ \
				f"Oringal Label: {true_label} Probability: {p_prob} Adversarial Label: {attack_pred}\n" + \
				f"Number of Substitutions {number_substituions} Sub rate: {sub_rate}\n" + \
				f"Attacks: {attack_list}\n\n"

			attack_report += report

			total_subs += number_substituions
			total_subrate += sub_rate
			total_prob += p_prob

		num_attacks = len(self.attack_examples)
		avearge_subs = total_subs/num_attacks if num_attacks != 0 else 0
		avearge_subrate = total_subrate/num_attacks if num_attacks != 0 else 0
		average_time = attack_time/num_attacks if num_attacks != 0 else 0
		average_prob = total_prob/num_attacks if num_attacks != 0 else 0

		general_report = f"Total Time: {attack_time} Average time: {average_time}\n" + \
			f"Total Queries {total_queries_mine} Avg Queries {avg_queries_mine}\n" +\
			f"Total PWWS Queries {total_queries_pwws} Avg PWWS Queries {avg_queries_pwws}\n" +\
			f"Adverarial Accuracy {new_accuracy}\n" + \
			f"Average Clean Probability {average_prob}\n" + \
			f"Total Substitutions: {total_subs} \nAverage Substitutions: {avearge_subs}\n" + \
			f"Average subrate {avearge_subrate}\n\n"
			

		return general_report, safe_report, attack_report


	def attack(self):

		start_time = time.time()

		#it would take 1 query each to get the probability for the predicted class
		total_examples = len(self.examples)
		total_queries_mine = total_examples
		total_queries_pwws = total_examples

		self._parse_sentences()

		self._get_all_synonyms()

		del(self.nlp)

		#we can infer the number of queries the original pwws would make for saliency and synonyms now
		#sum_s len(s) + sum_s sum_x len syn(s_x)
		total_queries_pwws += self._calc_pwws_queries()

		total_queries_mine += self._find_best_synonyms()

		total_queries_mine += self._calculate_saliancies()

		self._generate_multiword_attacks()

		total_queries_mine += self._peform_multiword_attacks()

		attack_time =  time.time() - start_time

		general_report, safe_report, attack_report  = self._generate_report(attack_time, total_queries_pwws, total_queries_mine)

		return general_report, safe_report, attack_report 

