# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10       #
#       Created on: 2023-01-15      #
# --------------------------------- #

import torch

class DisplayTrainingResults():

	def __init__(self, *arg, **kwargs):

		self.verbose = kwargs.pop("verbose", True)		
		self.summary_file = kwargs.pop("summary_file", None)

		log_file = kwargs.pop("log_file", None)
		self._open_logfile(log_file)

		self.max_train_buffer_length = kwargs.pop("max_train_buffer_length")

		self.display_keys = kwargs.pop("display_keys")

		self.train_buffer = []
		self._print_summary_header()


	def _open_logfile(self, log_file):

		self.log_stream = open(log_file, "a") if log_file is not None else None


	def _print_summary_header(self):

		if self.summary_file is not None:
			with open(self.summary_file, "a") as f:
				f.write(f"Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc\n")


	def flush_train_buffer(self):

		if len(self.train_buffer) > 0: 

			if self.verbose:

				output_last = self.train_buffer[-1]

				print('\r' + output_last, end = '\r')

			if self.log_stream is not None:

				output = '\n'.join(self.train_buffer) + '\n'

				self.log_stream.write(output)

			self.train_buffer = []


	def update_train_buffer(self, train_info):

		epoch = train_info[self.display_keys["epoch_num"]]
		batch_idx = train_info[self.display_keys["batch_idx"]]
		total_batches = train_info[self.display_keys["total_batches"]]

		loss = train_info[self.display_keys["train_loss"]]
		acc = train_info[self.display_keys["train_acc"]]
		run_time = train_info[self.display_keys["train_time"]]

		output_string = f"Epoch: {epoch} {batch_idx}/{total_batches} loss {loss:.4f}, acc {acc:.4f}, time {run_time:.2f}"

		self.train_buffer.append(output_string)

		if len(self.train_buffer) == self.max_train_buffer_length:
			self.flush_train_buffer()


	def print_inference_evalutation(self, inference_info):

		loss = inference_info[self.display_keys["inference_loss"]]
		acc = inference_info[self.display_keys["inference_acc"]]
		run_time = inference_info[self.display_keys["inference_time"]]

		output_string = f"\nInference Evaluation: loss {loss:.4f}, acc {acc:.4f}, time {run_time:.2f}\n"

		if self.verbose:
			print(output_string)

		if self.log_stream is not None:
			self.log_stream.write(output_string)
				

	def print_summary(self, train_info, inference_info):

		if self.summary_file is not None:
			with open(self.summary_file, "a") as f:				

				epoch = train_info[self.display_keys["epoch_num"]]

				train_loss = train_info[self.display_keys["train_loss"]]
				train_acc = train_info[self.display_keys["train_acc"]]

				val_loss = inference_info[self.display_keys["inference_loss"]]
				val_acc = inference_info[self.display_keys["inference_acc"]]

				f.write(f"{epoch}\t{train_loss:.4f}\t{train_acc:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n")


	def print_message(self, message):

		if self.verbose:
			print(message)

		if self.log_stream is not None:
			self.log_stream.write(message)
		



		







