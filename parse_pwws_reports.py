# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Sippola     #
#       Python Version: 3.7.1       #
#       Created on: 2021-06-12      #
# --------------------------------- #

"""
python3.10 parse_pwws_reports.py -attack_folder ../experiments/snli/ens_train_m/attacks
python3.10 parse_pwws_reports.py -attack_folder ../experiments/snli/ens_train_m_noise_det/attacks
python3.10 parse_pwws_reports.py -attack_folder ../experiments/snli/ens_train_mlp/attacks

python3.10 parse_pwws_reports.py -attack_folder ../experiments/snli_bilstm/attacks/
python3.10 parse_pwws_reports.py -attack_folder ../experiments/snli_roberta/attacks/
python3.10 parse_pwws_reports.py -attack_folder ../experiments/snli_deberta/attacks/
python3.10 parse_pwws_reports.py -attack_folder ../experiments/snli_bart/attacks/


python3.10 parse_pwws_reports.py -attack_folder ../experiments/snli_ensemble/attacks/
python3.10 parse_pwws_reports.py -attack_folder ../experiments/snli_ensemble_trans/attacks/
python3.10 parse_pwws_reports.py -attack_folder ../experiments/snli_ensemble_vote/attacks/
python3.10 parse_pwws_reports.py -attack_folder ../experiments/snli_ensemble_trans_vote/attacks/

python3.10 parse_pwws_reports.py -attack_folder ../experiments/snli_ensemble_trans_nd/attacks/
python3.10 parse_pwws_reports.py -attack_folder ../experiments/snli_ensemble_trans_noise/attacks/
"""

import argparse
import copy
import json
import os

data_indexes = [
	{"num_example":2, "number_attack":5, "number_safe":8},
	{"time":2, "avg_time":5},
	{"num_query":2, "avg_query":5},
	{"num_query_pwws":3, "avg_query_pwws":7},
	{"adv_acc":2},
	{"avg_class_prob":3},
	{"total_subs":2},
	{"avg_subs":2},
	{"avg_subrate":2}
]
		
def main(attack_folder):

	attack_folders = [os.path.splitext(file)[0] for file in os.listdir(attack_folder)]

	results = []
	for folder in attack_folders:

		result = {}

		file_tokes = folder.split("_")

		result |= {file_tokes[2*i]:file_tokes[2*i+1] for i in range(len(file_tokes)//2)}

		folder = os.path.join(attack_folder, folder)
		results_file = os.path.join(folder, "general.txt")
		
		with open(results_file, "r") as f:

			for line, indexes in zip(f, data_indexes):
				line = line.strip('\n').split()				
				result |= {key:line[idx] for key, idx in indexes.items()}

		if "nd" not in result:
			result["nd"] = 1

		results.append(result)



	results_per_budget = {}

	print(len(results))
	for result in results:
		bud = result["bud"]
		
		if bud not in results_per_budget:
			results_per_budget[bud] = {}			

		nd = result["nd"]	

		if nd not in results_per_budget[bud]:
			results_per_budget[bud][nd] = []

		results_per_budget[bud][nd].append(result)


	print(len(results_per_budget))
	for bud, nd_results in results_per_budget.items():
		for nd, results in nd_results.items():
			print(len(results))


	results_per_budget = {bud:{nd:sorted(result, key=lambda x:x["str"]) for nd, result in nd_results.items() } for bud, nd_results in results_per_budget.items()}

	for bud, nd_results in results_per_budget.items():

		for nd, results in nd_results.items():

			text_adv_acc = ""
			text_adv_acc_sheet = ""
			text_avg_class_prob = ""
			avg = []
			for r in results:

				adv_acc = float(r["adv_acc"])*100
				text_adv_acc += f" & {adv_acc:.1f}"
				text_adv_acc_sheet += f"{adv_acc:.1f}\t"

				avg_class_prob = float(r["avg_class_prob"])*100
				text_avg_class_prob += f"{avg_class_prob:.1f}\t"

				avg.append(adv_acc)

			avg = sum(avg)/len(avg)
			
			text_adv_acc += f" & {avg:.1f}"

			text_adv_acc += " \\\\"

			print()
			print(f"Budget: {bud} ND Factor {nd}")
			print()
			print(text_adv_acc)
			print()
			print(text_adv_acc_sheet)
			print()
			print(text_avg_class_prob)



			

		



if __name__ == '__main__':

	#parse command line
	parser = argparse.ArgumentParser()
	parser.add_argument("-attack_folder", help="template attack config file", required=True)
	args = parser.parse_args()

	main(args.attack_folder)