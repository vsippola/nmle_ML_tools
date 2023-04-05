'''

python3.10 ensemble-parser.py -input_file ../../../raw_data/nli_ens/nli_ens_full.txt -output_folder ../../parsed_data/ens/

'''

import argparse
import os
import pathlib


def parse_snli_file(input_path):

	full_corpus = []

	with open(input_path) as f:

		header = f.readline() #incase we want this later?

		for _, line in enumerate(f):

			line_tokens = line.split('\t')

			label = line_tokens[0]
	
			s1 = line_tokens[5]
			s2 = line_tokens[6]

			if label != "-":

				full_corpus.append([label, s1, s2])

	return full_corpus


def corpus_by_label(full_corpus):

	full_corpus_label = {}

	for example in full_corpus:

		label = example[0]

		if label not in full_corpus_label:
			full_corpus_label[label] = []

		full_corpus_label[label].append(example)

	return full_corpus_label


def divide_corpus_label(full_corpus_label, dev_size):

	per_label_size = dev_size/len(full_corpus_label)

	train_corpus = []
	dev_corpus = []

	for _, corpus_label in full_corpus_label.items():

		label_threshold = len(corpus_label)/per_label_size
		acc = 0.0

		for example in corpus_label:
			if acc > label_threshold:
				dev_corpus.append(example)
				acc -= label_threshold
			else:
				train_corpus.append(example)

			acc+=1

	train = []
	dev = []
	for text_coprus, corpus in zip([train, dev],[train_corpus, dev_corpus]):
		for example_num, example in enumerate(corpus):
			example_text = f"{example_num}\t{example[0]}\t{example[1]}\t{example[2]}\n"
			text_coprus.append(example_text)

	train = "".join(train)
	dev = "".join(dev)

	return train, dev



def save_files(corpora, files, output_folder):

	for corpus, file in zip(corpora, files):
	
		if not (os.path.isdir(output_folder)):
			print()
			print(f'folder {output_folder} does not exist creating it')
			pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

		output_path = os.path.join(output_folder, file)

		with open(output_path, "w") as f:
			f.write(corpus)


def process_snli_file(input_path, output_folder, dev_size):

	#check input file
	if not (os.path.isfile(input_path)):
		print()
		print(f'file {input_path} does not exist')
		return False

	full_corpus = parse_snli_file(input_path)

	full_corpus_label = corpus_by_label(full_corpus)

	train, dev = divide_corpus_label(full_corpus_label,dev_size)

	save_files([train, dev], ["ens_train.tsv","ens_dev.tsv"], output_folder)

	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-input_file", help="input path to snli corpus", required=True)
	parser.add_argument("-output_folder", help="output path to parsed text corpus", required=True)
	parser.add_argument("-dev_size", help="output path to parsed text corpus", type=int, default=5000)
	args = parser.parse_args()

	process_snli_file(args.input_file, args.output_folder, args.dev_size)


if __name__ == '__main__':
	main()
