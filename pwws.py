# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.7.1       #
#       Created on: 2021-06-12      #
# --------------------------------- #

"""
python3.10 pwws.py -config_template_file "../configs/attacks/bilstm/template.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 1 5 50

python3.10 pwws.py -config_template_file "../configs/attacks/bart/template_safer.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 5
python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_trans_train_noise/template_safer.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 5




python3.10 pwws.py -config_template_file "../configs/attacks/roberta/template.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 1 5
python3.10 pwws.py -config_template_file "../configs/attacks/deberta/template.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 1 5
python3.10 pwws.py -config_template_file "../configs/attacks/bart/template.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 1 5
python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_trans/template.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 1 5
python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_trans_train/template.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 1 5
python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_trans_train_noise/template.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 1 5


python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_trans_train/template.json" -attack_size 250 -attack_strengths 0 1 2 3 4 5 6 7 8 9 -attack_budgets 1 5
python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_trans_train/template.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 50

python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_trans_train_noise/template.json" -attack_size 250 -attack_strengths 0 1 2 3 4 5 6 7 8 9 -attack_budgets 1 5
python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_trans_train_noise/template.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 50



python3.10 pwws.py -config_template_file "../configs/attacks/roberta/template.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 50
python3.10 pwws.py -config_template_file "../configs/attacks/deberta/template.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 50
python3.10 pwws.py -config_template_file "../configs/attacks/bart/template.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 50
python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_trans/template.json" -attack_size 10000 -attack_strengths 0 -attack_budgets 50


python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_trans_nd/template.json" -attack_size 250 -attack_strengths 0 1 2 3 4 5 6 7 8 9 -attack_budgets 1 5 -nd_values 1 2 4 8
python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_trans_noise/template.json" -attack_size 250 -attack_strengths 0 1 2 3 4 5 6 7 8 9 -attack_budgets 1 5 -nd_values 1 2 4 8

python3.10 pwws.py -config_template_file "../configs/attacks/ensemble/template.json" -attack_size 250 -attack_strengths 0 1 2 3 4 5 6 7 8 9 -attack_budgets 1 5
python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_trans/template.json" -attack_size 250 -attack_strengths 0 1 2 3 4 5 6 7 8 9 -attack_budgets 1 5
python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_vote/template.json" -attack_size 250 -attack_strengths 0 1 2 3 4 5 6 7 8 9 -attack_budgets 1 5
python3.10 pwws.py -config_template_file "../configs/attacks/ensemble_trans_vote/template.json" -attack_size 250 -attack_strengths 0 1 2 3 4 5 6 7 8 9 -attack_budgets 1 5


python3.10 pwws.py -config_template_file "../configs/attacks/bilstm/template.json" -attack_size 250 -attack_strengths 0 1 2 3 4 5 6 7 8 9 -attack_budgets 1 5
python3.10 pwws.py -config_template_file "../configs/attacks/roberta/template.json" -attack_size 250 -attack_strengths 0 1 2 3 4 5 6 7 8 9 -attack_budgets 1 5
python3.10 pwws.py -config_template_file "../configs/attacks/deberta/template.json" -attack_size 250 -attack_strengths 0 1 2 3 4 5 6 7 8 9 -attack_budgets 1 5
python3.10 pwws.py -config_template_file "../configs/attacks/bart/template.json" -attack_size 250 -attack_strengths 0 1 2 3 4 5 6 7 8 9 -attack_budgets 1 5


"""

import argparse
import copy
import json
import os

from attack_code.do_pwws_attack import do_pwws


def get_info(attack_strengths, attack_size, attack_budgets, nd_values):

	nd_i = 0 if len(nd_values) > 0 else -1

	while nd_i < len(nd_values):


		if len(nd_values) == 0:
			nd_value = None
			nd_suffix = ""

		else:

			nd_value = nd_values[nd_i]
			nd_suffix = f"_nd_{nd_value}"

		for strength in attack_strengths:

			for budget in attack_budgets:

				output_folder_prefix = f"size_{attack_size}_str_{strength}_bud_{budget}{nd_suffix}"

				yield strength, budget, nd_value, output_folder_prefix


		nd_i += 1


def main(template_config, attack_strengths, attack_size, attack_budgets, nd_values):

	base_folder = template_config["pwws_config"]["output_folder"]

	template_config["attack_corpus_params"]["number_of_examples"] = attack_size

	for strength, budget, nd_value, output_folder_prefix in get_info(attack_strengths, attack_size, attack_budgets, nd_values):

		tempate_copy = copy.deepcopy(template_config)

		tempate_copy["attack_corpus_params"]["attack_strength"] = strength

		if nd_value is not None:

			tempate_copy["pwws_config"]["nondet_factor"] = nd_value

		tempate_copy["pwws_config"]["attack_budget"] = budget
		tempate_copy["pwws_config"]["output_folder"] = os.path.join(base_folder, output_folder_prefix)

		do_pwws(tempate_copy)
		

if __name__ == '__main__':

	#parse command line
	parser = argparse.ArgumentParser()
	parser.add_argument("-config_template_file", help="template attack config file", required=True)

	parser.add_argument("-attack_strengths", help="template attack config file", required=True, nargs='+', type=int)

	parser.add_argument("-attack_budgets", help="template attack config file", required=True, nargs='+', type=int)

	parser.add_argument("-attack_size", help="template attack config file", required=True, type=int)

	parser.add_argument("-nd_values", help="template attack config file", required=False, nargs='+', type=int, default=[])

	args = parser.parse_args()

	with open(args.config_template_file, 'r') as f:
		template_config = json.load(f)

	main(template_config, args.attack_strengths, args.attack_size,  args.attack_budgets, args.nd_values)