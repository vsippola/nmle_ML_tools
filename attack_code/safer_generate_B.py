# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.7.1       #
#       Created on: 2021-06-12      #
# --------------------------------- #

"""
python3.10 attack_code/safer_generate_B.py  -synonym_json_file ../safer/counterfitted_neighbors.json -B_json_file ../safer/B_sets_mine_2.json -hops 2
python3.10 attack_code/safer_generate_B.py  -synonym_json_file ../safer/counterfitted_neighbors.json -B_json_file ../safer/B_sets_mine_3.json -hops 3

python3.10 attack_code/safer_generate_B.py  -synonym_json_file ../safer/counterfitted_neighbors.json -B_json_file ../safer/B_sets.json -paper_method True

"""

import argparse
import json
import os
import pathlib
import sys


#uses transitive property of bsets regarding synonyms to generate bsets
def paper_generate_B_sets(synonyms):

	needs_B_set = []

	for word, syns in synonyms.items():

		if len(syns) > 0:

			needs_B_set.append([word]+syns)

	B_sets = {}
	word2bset = {}
	num_b_sets = 0	
	
	for syn_set in needs_B_set:

		existing_b_sets = list(set([word2bset[w] for w in syn_set if w in word2bset]))

		#if there is no existing b_sets for any words in this syn set
		if len(existing_b_sets) == 0:

			b_set_idx = str(num_b_sets)
			num_b_sets += 1

			for word in syn_set:
				word2bset[word] = b_set_idx

			B_sets[b_set_idx] = set(syn_set)

		#otherwise there is at least one existing b_set for a word in the synset
		#so we need to add these words to this set
		#and if there is more than one existing b_Set, merge them
		else:

			b_set_idx = existing_b_sets.pop()			

			merge_b_sets = [set(syn_set)] + [B_sets.pop(bs_i) for bs_i in existing_b_sets]

	
			for old_b_set in merge_b_sets:

				for word in old_b_set:
					word2bset[word] = b_set_idx

				B_sets[b_set_idx] = B_sets[b_set_idx].union(old_b_set)

	B_sets = {bs_i:list(b_set) for bs_i, b_set in B_sets.items()}

	test = sorted([len(bset) for _, bset in B_sets.items()])

	#print(test)

	B = {
		"sets":B_sets,
		"w2set":word2bset
	}

	return B


def my_generate_B_setS(synonyms, hops):

	#filter words without a bset, and get 1 hop Bset
	needs_B_set = {}
	for word, syns in synonyms.items():

		if len(syns) > 0:

			needs_B_set[word] = set(syns)

	display_count = 50

	b_sets = {}

	for w_i, word in enumerate(needs_B_set):

		searching = True

		h = 1
		checked = {}

		b_sets[word] = needs_B_set[word]

		while (h < hops) and (searching):

			new_words = set([])

			for b_word in b_sets[word]:

				if b_word not in checked:
					checked[b_word] = None

					new_words = new_words.union(needs_B_set[b_word])

			if len(new_words) == 0:
				searching = False

			else:
				b_sets[word] = b_sets[word].union(new_words)

			h+=1
		if (w_i%display_count) == 0:
			print(f"\r {w_i+1}/{len(needs_B_set)}  ", end='\r')

	print(f"\r {w_i+1}/{len(needs_B_set)}  ")
	print()
	print()

	B_sets = {}
	word2bset = {}

	for word, bset in b_sets.items():
		word2bset[word] = str(len(B_sets))
		B_sets[str(len(B_sets))] = list(bset)

	test = sorted([len(bset) for _, bset in B_sets.items()])

	#print(test)

	B = {
		"sets":B_sets,
		"w2set":word2bset
	}

	return B


def old_main(synonyms, B_file):

	B = paper_generate_B_sets(synonyms)

	output_folder = os.path.dirname(B_file)

	pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

	with open(B_file, "w") as f:
		json.dump(B, f)


def	main(synonyms, hops, B_file):

	B = my_generate_B_setS(synonyms, hops)

	output_folder = os.path.dirname(B_file)

	pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

	with open(B_file, "w") as f:
		json.dump(B, f)

if __name__ == '__main__':

	#parse command line
	parser = argparse.ArgumentParser()
	parser.add_argument("-synonym_json_file", help="path to counterfitted synonyms", required=True)
	parser.add_argument("-B_json_file", help="path to B sets as defined in paper", required=True)
	parser.add_argument("-paper_method", help="use raw method from paper", required=False, type=bool, default=False)
	parser.add_argument("-hops", help="if not using paper how many synonym hops to do", required=False, type=int, default=3)
	args = parser.parse_args()

	with open(args.synonym_json_file, 'r') as f:
		synonyms = json.load(f)

	if args.paper_method:

		old_main(synonyms, args.B_json_file)

	else:

		main(synonyms, args.hops, args.B_json_file)

