'''
parses the semeval sentence similarity dataset

python3.10 word2sense_pickle.py -input_file ../../../raw_data/word2sense/Word2Sense.txt -output_folder ../../word2vec_pickles/word2sense/
'''

import argparse
import os
import pathlib
import pickle

import numpy as np

def parse_word2sense_textfile(input_path):

	PRINT_STEP = 2500

	vocab = {"__OOV":0}
	vecs = [0]

	with open(input_path) as f:

		for line_num, line in enumerate(f):
			
			line_tokens = line.split()
			
			word = line_tokens[0]
			vec = np.array(line_tokens[1:]).astype(np.float32)

			vocab[word] = len(vecs)
			vecs.append(vec)


			if (line_num+1) % PRINT_STEP == 0:
				print(f'\rlines processed: {line_num+1}', end='\r')

	vecs[0] = np.zeros(vecs[1].shape, dtype=np.float32)

	vecs = np.stack(vecs)

	print()
	print()

	return vocab, vecs


def save_files(vocab, vecs, output_folder):

	#create output folder if it doesn't exist
	if not (os.path.isdir(output_folder)):
		print()
		print(f'folder {output_folder} does not exist creating it')
		pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

	vocab_path = os.path.join(output_folder, 'vocab.pkl')
	vecs_path = os.path.join(output_folder, 'vecs.pkl')

	with open(vocab_path, 'wb') as f:
		pickle.dump(vocab, f)

	with open(vecs_path, 'wb') as f:
		pickle.dump(vecs, f)


def process_word2sense_textfile(input_path, output_folder):

	#check input file
	if not (os.path.isfile(input_path)):
		print()
		print(f'file {input_path} does not exist')
		return False

	vocab, vecs = parse_word2sense_textfile(input_path)

	save_files(vocab, vecs, output_folder)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-input_file", help="input path to the text word2sense file", required=True)
	parser.add_argument("-output_folder", help="output path to save pickle files (vectors and vocab)", required=True)
	args = parser.parse_args()

	process_word2sense_textfile(args.input_file, args.output_folder)


if __name__ == '__main__':
	main()
