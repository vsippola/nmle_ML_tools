'''
parses the semeval sentence similarity dataset

either run from command line script or call process_sts_file in python code

python3.10 semeval-sts-parser.py -input <input path> -output <output path>


python3.10 semeval-sts-parser.py -input_file ../../../raw_data/stsbenchmark/sts-dev.csv -output_file ../../parsed_data/stsbenchmark/sts-dev.csv
python3.10 semeval-sts-parser.py -input_file ../../../raw_data/stsbenchmark/sts-test.csv -output_file ../../parsed_data/stsbenchmark/sts-test.csv
python3.10 semeval-sts-parser.py -input_file ../../../raw_data/stsbenchmark/sts-train.csv -output_file ../../parsed_data/stsbenchmark/sts-train.csv
'''

import argparse
import os

from multiprocessing import cpu_count
from stanza.server import CoreNLPClient

class stsparser_constants:
	ANNOTATION = 'tokenize'
	TIMEOUT = 30000
	MEMORY = '1G'
	THREADS = cpu_count() - 1
	OUTPUT_FORMAT = 'json'
	QUIET = True
	LANG = 'english'

def parse_sts_file(input_path):

	#start the parsing client
	corenlp_client = CoreNLPClient(
			annotators=stsparser_constants.ANNOTATION,
			timeout=stsparser_constants.TIMEOUT,
			memory=stsparser_constants.MEMORY,
			threads=stsparser_constants.THREADS,
			output_format=stsparser_constants.OUTPUT_FORMAT,
			be_quiet=stsparser_constants.QUIET
		)

	prased_file = ''

	with open(input_path) as f:

		for example_num, line in enumerate(f):

			line_tokens = line.split('\t')

			label = line_tokens[4]
			sents = [line_tokens[5], line_tokens[6]]

			for s_i in range(2):
				sents[s_i] = corenlp_client.annotate(sents[s_i], properties=stsparser_constants.LANG) 
				sents[s_i] = [token['word'] for token in sents[s_i]['tokens']]
				sents[s_i] = ' '.join(sents[s_i])


			parsed_example = f'{example_num}\t{label}\t{sents[0]}\t{sents[1]}\n'

			prased_file += parsed_example

	corenlp_client.stop()

	return prased_file


def save_file(parsed_sts_file, output_path):

	#create output folder if it doesn't exist
	output_dir = os.path.dirname(output_path)
	print(output_dir)
	if not (os.path.isdir(output_dir)):
		print()
		print(f'folder {output_dir} does not exist creating it')
		os.makedirs(output_dir)

	with open(output_path, "w") as f:
		f.write(parsed_sts_file)



def process_sts_file(input_path, output_path):

	#check input file
	if not (os.path.isfile(input_path)):
		print()
		print(f'file {input_path} does not exist')
		return False

	parsed_sts_file = parse_sts_file(input_path)

	save_file(parsed_sts_file, output_path)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-input_file", help="input path to sts corpus", required=True)
	parser.add_argument("-output_file", help="output path to parsed text corpus", required=True)
	args = parser.parse_args()

	process_sts_file(args.input_file, args.output_file)


if __name__ == '__main__':
	main()
