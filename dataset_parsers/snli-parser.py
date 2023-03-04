'''



python3.10 snli-parser.py -input_file ../../../raw_data/snli_1.0/snli_1.0/snli_1.0_dev.txt -output_file ../../parsed_data/snli_1.0/snli_1.0_dev.tsv
python3.10 snli-parser.py -input_file ../../../raw_data/snli_1.0/snli_1.0/snli_1.0_test.txt -output_file ../../parsed_data/snli_1.0/snli_1.0_test.tsv
python3.10 snli-parser.py -input_file ../../../raw_data/snli_1.0/snli_1.0/snli_1.0_train.txt -output_file ../../parsed_data/snli_1.0/snli_1.0_train.tsv

'''

import argparse
import os


def parse_sts_file(input_path):

	prased_file = ''

	with open(input_path) as f:

		header = f.readline() #incase we want this later?

		for example_num, line in enumerate(f):

			line_tokens = line.split('\t')

			label = line_tokens[0]

			sents = [line_tokens[1], line_tokens[2]]

			for s_i, sent in enumerate(sents):
				sents_tokens = sent.replace('(', '').replace(')', '').split()
				sents[s_i] = ' '.join(sents_tokens)

			parsed_example = f'{example_num}\t{label}\t{sents[0]}\t{sents[1]}\n'

			prased_file += parsed_example

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
