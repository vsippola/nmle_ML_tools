'''
parses the semeval sentence similarity dataset

python3.10 make_training_files.py -corpus_dir "../../parsed_data/snli_1.0/" -corpus_files "snli_1.0_dev.tsv" "snli_1.0_test.tsv" "snli_1.0_train.tsv" -sentence_indexes 2 3 -input_folder ../../word2vec_pickles/fasttext_crawl-300d-2M/ -output_folder ../../word2vec_pickles/fasttext_crawl-300d-2M/snli_1.0/ 

python3.10 make_training_files.py -corpus_dir "../../parsed_data/snli_1.0/" -corpus_files "snli_1.0_dev.tsv" "snli_1.0_test.tsv" "snli_1.0_train.tsv" -sentence_indexes 2 3 -input_folder ../../word2vec_pickles/word2sense/ -output_folder ../../word2vec_pickles/word2sense/snli_1.0/ 
'''

import argparse
import os
import pathlib
import pickle

from nltk.tokenize import word_tokenize
import numpy as np

def make_corpus_files(corpus_dir, files):

	corpus_files = []

	for file in files:
		corpus_file = os.path.join(corpus_dir, file)		
		corpus_files.append(corpus_file)

	return corpus_files


def load_w2v(input_folder):

	w2v=[]
	for file in ["vocab.pkl", "vecs.pkl"]:
		file_path = os.path.join(input_folder, file)
		with open(file_path, "rb") as f:
			w2v_file = pickle.load(f)
			w2v.append(w2v_file)

	return w2v[0], w2v[1]


def get_corpus_vocab(corpus_files, sentence_indexes):

	PRINT_STEP = 2500
	line_count = 0

	corpus_vocab = {}

	for corpus_file in corpus_files:
		with open(corpus_file, "r") as f:

			for line in f:

				line_tokens = line.strip().split('\t')

				for s_i in sentence_indexes:

					sentence = line_tokens[s_i]

					word_tokens = word_tokenize(sentence)

					for word in word_tokens:
						
						corpus_vocab[word] = None

				if (line_count+1) % PRINT_STEP == 0:
					print(f'\rlines processed: {line_count+1}', end='\r')

				line_count += 1

	print()
	print()
	print(f"Unique Corpus Tokens: {len(corpus_vocab)}")

	return corpus_vocab


def filter_w2v(vocab, vecs, corpus_vocab):

	OOV_i = 0
	
	f_vocab = {"__OOV":OOV_i}
	f_vecs = [vecs[vocab["__OOV"]]]

	for word in corpus_vocab:

		if word not in vocab:
			f_vocab[word] = OOV_i

		else:
			f_vocab[word] = len(f_vecs)
			f_vecs.append(vecs[vocab[word]])

	f_vecs = np.stack(f_vecs)

	return f_vocab, f_vecs


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

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-corpus_dir", help="input path to the corpus files", required=True)
	parser.add_argument("-corpus_files", help="tab seperated corpus files", required=True, nargs='+')
	parser.add_argument("-sentence_indexes", help="sentence indexes", required=True, type=int, nargs='+')
	parser.add_argument("-input_folder", help="folder containing w2v vec and vocab", required=True)
	parser.add_argument("-output_folder", help="folder to put filterd w2v vecs and vocab", required=True)

	args = parser.parse_args()

	corpus_files = make_corpus_files(args.corpus_dir, args.corpus_files)

	corpus_vocab = get_corpus_vocab(corpus_files, args.sentence_indexes)

	vocab, vecs = load_w2v(args.input_folder)

	f_vocab, f_vecs = filter_w2v(vocab, vecs, corpus_vocab)

	save_files(f_vocab, f_vecs, args.output_folder)


if __name__ == '__main__':
	main()
