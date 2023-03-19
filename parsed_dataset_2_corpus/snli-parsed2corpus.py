'''
parses the semeval sentence similarity dataset

python3.10 snli-parsed2corpus.py -input_folder ../../parsed_data/snli_1.0/ -output_folder ../../corpus_data/snli_1.0_w2sense/ -word2vec_folder ../../word2vec_pickles/word2sense/
python3.10 snli-parsed2corpus.py -input_folder ../../parsed_data/snli_1.0/ -output_folder ../../corpus_data/snli_1.0_fasttext/ -word2vec_folder ../../word2vec_pickles/fasttext_crawl-300d-2M/

'''

import argparse
import os
import pickle
import numpy as np

class sts_parsed2corpus_const:
	
	CORUPUS_OUTPUT = {'snli_1.0_dev.tsv':'dev.pkl', 'snli_1.0_test.tsv':'test.pkl', 'snli_1.0_train.tsv':'train.pkl'}
	CORPUS_FILES = ['snli_1.0_dev.tsv', 'snli_1.0_test.tsv', 'snli_1.0_train.tsv']
	W2V_VOCAB_FILE = 'vocab.pkl'
	W2V_VECS_FILE = 'vecs.pkl'
	LABEL_TEXT_2_NUM = {'entailment':0, 'neutral':1, 'contradiction':2}


def get_corpus_vocab(input_folder):

	corpus_vocab = {}

	for file in sts_parsed2corpus_const.CORPUS_FILES:

		file_path = os.path.join(input_folder, file)

		with open(file_path) as f:

			for line in f:

				line_tokens = line[:-1].split('\t')				

				sents = [line_tokens[2], line_tokens[3]]

				for sent in sents:
					words = sent.split(' ')

					# OOV words will be mapped to 0 index 0 vector
					for word in words:
						if word not in corpus_vocab:
							corpus_vocab[word] = len(corpus_vocab) + 1

	return corpus_vocab


def get_w2v(word2vec_folder):

	w2v_vocab_file = os.path.join(word2vec_folder, sts_parsed2corpus_const.W2V_VOCAB_FILE)

	with open(w2v_vocab_file, 'rb') as f:
		w2v_vocab = pickle.load(f)


	w2v_vecs_file = os.path.join(word2vec_folder, sts_parsed2corpus_const.W2V_VECS_FILE)

	with open(w2v_vecs_file, 'rb') as f:
		w2v_vecs = pickle.load(f)


	return w2v_vocab, w2v_vecs


def generate_corpus_w2v(corpus_vocab, w2v_vocab, w2v_vec):

	vec_len = len(w2v_vec[0])
	corpus_vecs = [np.zeros(vec_len, dtype=np.float32)]

	count = 0
	oov_count = 0

	for word in corpus_vocab:

		#if the word is in the w2v
		if word in w2v_vocab:
			
			#add the w2v vector to the corpus vectors
			vec_index = w2v_vocab[word]
			vec = w2v_vec[vec_index]
			corpus_vecs.append(vec)

			#udpate the coprus w2v index to adjust for OOV
			corpus_vocab[word] = corpus_vocab[word] - oov_count


		#otherwise the word is OOV and is set to 0
		else:
			corpus_vocab[word] = 0
			oov_count += 1

	corpus_vecs = np.asarray(corpus_vecs, dtype=np.float32)

	return corpus_vocab, corpus_vecs



def generate_corpus_files(input_folder, corpus_vocab):

	corpus_files = {}

	for file in sts_parsed2corpus_const.CORPUS_FILES:

		corpus_files[file] = []

		file_path = os.path.join(input_folder, file)

		with open(file_path) as f:

			for line in f:
				
				line_tokens = line[:-1].split('\t')	

				example_num = int(line_tokens[0])

				#evaluate datasts may have a '-' gold label that should be ignored during out evaluation
				if line_tokens[1] in sts_parsed2corpus_const.LABEL_TEXT_2_NUM:
					label = sts_parsed2corpus_const.LABEL_TEXT_2_NUM[line_tokens[1]]

					text_sents = [line_tokens[2],line_tokens[3]]
					index_sents = []


					for sent in text_sents:

						words = sent.split(' ')
						index_sent = [corpus_vocab[word] for word in words]
						index_sents.append(index_sent)

					example = [example_num, label, index_sents[0], index_sents[1]]

					corpus_files[file].append(example)

	return corpus_files



def save_files(corpus_files, corpus_vocab, corpus_vecs, output_folder):

	#create output folder if it doesn't exist
	if not (os.path.isdir(output_folder)):
		print()
		print(f'folder {output_folder} does not exist creating it')
		os.makedirs(output_folder)


	#save corpus files
	for file in corpus_files:

		file_name = os.path.join(output_folder, sts_parsed2corpus_const.CORUPUS_OUTPUT[file])

		with open(file_name, 'wb') as f:
			pickle.dump(corpus_files[file], f)

	#save corpus vocab
	file_name = os.path.join(output_folder, sts_parsed2corpus_const.W2V_VOCAB_FILE)
	with open(file_name, 'wb') as f:
		pickle.dump(corpus_vocab, f)

	#save corpus vectors
	file_name = os.path.join(output_folder, sts_parsed2corpus_const.W2V_VECS_FILE)
	with open(file_name, 'wb') as f:
		pickle.dump(corpus_vecs, f)
		


def creat_sts_corpus(input_folder, output_folder, word2vec_folder):

	if not (os.path.isdir(input_folder)):
		print()
		print(f'folder {input_folder} does not exist')
		return False

	if not (os.path.isdir(word2vec_folder)):
		print()
		print(f'folder {word2vec_folder} does not exist')
		return False


	corpus_vocab = get_corpus_vocab(input_folder)
	w2v_vocab, w2v_vecs = get_w2v(word2vec_folder)

	corpus_vocab, corpus_vecs = generate_corpus_w2v(corpus_vocab, w2v_vocab, w2v_vecs)

	corpus_files = generate_corpus_files(input_folder, corpus_vocab)

	save_files(corpus_files, corpus_vocab, corpus_vecs, output_folder)





def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-input_folder", help="input path to parsed sts files", required=True)
	parser.add_argument("-output_folder", help="output path to save pickle files and vocab", required=True)
	parser.add_argument("-word2vec_folder", help="output path to save pickle files and vocab", required=True)
	args = parser.parse_args()

	creat_sts_corpus(args.input_folder, args.output_folder,args.word2vec_folder)


if __name__ == '__main__':
	main()
