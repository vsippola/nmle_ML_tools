# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10      #
#       Created on: 2023-01-28      #
# --------------------------------- #

import os
import time

import torch

class ModelTrainer:

	def __init__(self, *args, **kwargs):
		
		self.training_display_keys = kwargs.pop("training_display_keys")		
		
		self.training_info = {
			"max_epochs":kwargs.pop("max_epochs", None),
			"current_epoch":0,
			"max_rounds":kwargs.pop("max_rounds", 1),
			"current_round":0,
			"tenacity":kwargs.pop("tenacity", 7),
			"current_fail_count":0,
			"minimum_lr":kwargs.pop("minimum_lr", 1e-7),			
			"best_inference_acc":-float("inf"),
			"best_checkpoint_filename":None,
			"save_all":kwargs.pop("save_all", False),
			"checkpoint":0,
			"check_after_iterations":kwargs.pop("check_after_iterations", None),
			"current_iterations":0
		}		

		self.checkpoint_folder = kwargs.pop("checkpoint_folder")

		self.load_from_checkpoint = kwargs.pop("load_from_checkpoint", None)		

		self.batch_loss_key = kwargs.pop("batch_loss_key")
		self.inference_acc_key = kwargs.pop("inference_acc_key")

		#optimzier/scheduler TODO

		self.dataloader = None
		self.metric_tracker = None
		self.training_displayer = None	
		self.inference_evaluator = None
		self.optimizer = None


	def set_dataloader(self, dataloader):
		self.dataloader = dataloader

	def set_metric_tracker(self, metric_tracker):
		self.metric_tracker = metric_tracker

	def set_training_displayer(self, training_displayer):
		self.training_displayer = training_displayer

	def set_inference_evaluator(self, inference_evaluator):
		self.inference_evaluator = inference_evaluator

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer


	def _load_checkpoint(self, model):

		checkpoint_filename = os.path.join(self.checkpoint_folder, self.load_from_checkpoint)

		checkpoint = torch.load(checkpoint_filename)

		model.load_from_dict(**checkpoint["model_params"])
		self.optimizer.load_from_dict(**checkpoint["optimizer_params"])
		self.training_info = checkpoint["training_info"]

		self.training_info["current_epoch"] += 1


	def _evaluate_inference(self, model, training_display_info):

		current_checkpoint = self.training_info["checkpoint"]
		training_display_info["checkpoint"] = current_checkpoint

		inferecene_info = self.inference_evaluator(model)

		self.training_displayer.print_inference_evalutation(inferecene_info)

		self.training_displayer.print_summary(training_display_info, inferecene_info)

		new_acc = inferecene_info[self.inference_acc_key]
		best_acc = self.training_info["best_inference_acc"]
		
		checkpoint_filename = os.path.join(self.checkpoint_folder, f"Checkpoint_{current_checkpoint}.tar")

		self.training_info["improved"] = False

		if new_acc >= best_acc:
			self.training_info["best_inference_acc"] = new_acc
			self.training_info["best_checkpoint_filename"] = checkpoint_filename
			self.training_info["improved"] = True
			
			self.training_displayer.print_message(f"\nNew best acc: {new_acc}\n")			

		if self.training_info["save_all"] or self.training_info["improved"]:

			self.training_displayer.print_message(f"\nSaving Checkpoint...\n\n")		

			checkpoint = {
				"model_params":model.get_param_dict(),
				"optimizer_params":self.optimizer.get_param_dict(),
				"training_info":self.training_info
			}
			torch.save(checkpoint, checkpoint_filename)

		self.training_info["checkpoint"] += 1

		self._evaluate_training(model)

		model.train()


	def _evaluate_training(self, model):

		#if there is a maximum number of epochs stop if we have reached it
		max_epochs = self.training_info["max_epochs"]
		curr_epoch = self.training_info["current_epoch"] 

		if (max_epochs is not None)  and (curr_epoch >= max_epochs):
			self.training_displayer.print_message("\n Last epoch reached. Ending Training \n")
			self.training_info["continue_training"] = False
			return		

		#if we improved this round no need to stop training
		improved = self.training_info["improved"]

		if improved:
			self.training_info["continue_training"] = True
			self.training_info["current_fail_count"] = 0
			return

		#otherwise we failed to improve so increase the count
		self.training_info["current_fail_count"] += 1

		#check if we have reach the fail state
		curr_fail_count = self.training_info["current_fail_count"]
		tenacity = self.training_info["tenacity"]

		#we can still fail more times
		if curr_fail_count < tenacity:
			self.training_info["continue_training"] = True
			return

		#otherwise we have reached the fail state for this round of training
		#increase the round count
		self.training_info["current_round"] += 1
		self.training_info["current_fail_count"] = 0

		#check if we have lost all the rounds
		curr_round = self.training_info["current_round"]
		max_rounds = self.training_info["max_rounds"]

		if curr_round >= max_rounds:
			self.training_displayer.print_message("\n Last round reached. Ending Training \n")
			self.training_info["continue_training"] = False
			return	

		#if we are already at the minmum learning rate then restating trianing should not help
		min_lr = self.training_info["minimum_lr"]
		curr_lr = self.optimizer.current_lr()

		if (curr_lr is not None) and (curr_lr <= min_lr):
			self.training_displayer.print_message("\n Minimum Learning rate reached. Ending Training \n")
			self.training_info["continue_training"] = False
			return	


		self.training_displayer.print_message(f"\n Round {curr_round+1} of {max_rounds} starting\n")
		self.training_displayer.print_message(f"\n Loading Best Checkpoint... \n")

		best_checkpoint_fname = self.training_info["best_checkpoint_filename"]

		best_checkpoint = torch.load(best_checkpoint_fname)

		model.load_from_dict(**best_checkpoint["model_params"])
		self.optimizer.load_from_dict(**best_checkpoint["optimizer_params"])

		#step until we are at a new learning rate
		while self.optimizer.current_lr() >= curr_lr:
			self.optimizer.scheduler_step()			

		self.training_displayer.print_message(f"\n New Learning Rate:{self.optimizer.current_lr()}\n")

		self.training_info["continue_training"] = True


	def __call__(self, model):		

		self.optimizer.register_model(model)

		if self.load_from_checkpoint is not None:
			self._load_checkpoint(model)

		training_display_info = {self.training_display_keys["total_batches"]:len(self.dataloader)}

		self.training_info["continue_training"] = True
		self.training_info["batch_size"] = self.dataloader.batch_size

		while self.training_info["continue_training"]:

			training_display_info[self.training_display_keys["epoch_num"]] = self.training_info["current_epoch"] + 1			

			self.metric_tracker.reset_metrics()

			epoch_start = time.time()

			model.train()

			for batch_idx, batch_data in enumerate(self.dataloader):

				training_display_info[self.training_display_keys["batch_idx"]] = batch_idx + 1
	
				self.optimizer.zero_grad(set_to_none=True)

				state_object = model(batch_data)
				self.metric_tracker.update_metrics(state_object)

				loss = state_object[self.batch_loss_key]
				loss.backward()

				with torch.no_grad():

					# Optimization step - apply the gradients
					self.optimizer.optimizer_step()

					train_metrics = self.metric_tracker.get_metrics()
					
					# Record end time for the batch
					training_display_info[self.training_display_keys["time"]] = (time.time() - epoch_start)/60.0		

					training_display_info = training_display_info | train_metrics
						
					self.training_displayer.update_train_buffer(training_display_info)

					if self.training_info["check_after_iterations"] is not None:

						self.training_info["current_iterations"] += self.training_info["batch_size"]

						if self.training_info["current_iterations"] >= self.training_info["check_after_iterations"]:

							self.training_info["current_iterations"]  = 0
							self._evaluate_inference(model, training_display_info)															

							if not self.training_info["continue_training"]:
								break


			self.training_displayer.flush_train_buffer()				

			if not self.training_info["continue_training"]:
				break

			self.training_info["current_iterations"]  = 0
			self._evaluate_inference(model, training_display_info)				

			self.training_info["current_epoch"] += 1
