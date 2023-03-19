# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

import sys

from .sentence_list_builder import SentenceListDatasetBuilder



class DatasetFactory():

	BUILDER_CLASS_TYPE = {
		"sentence_list": SentenceListDatasetBuilder
	}

	DATASET_CLASS_TYPE = {
	
	}

	dataset_builders = {dataset_type:None for dataset_type in BUILDER_CLASS_TYPE}


	@classmethod
	def BUILD_DATASET(cls, *args, **kwargs):

		dataset_type = kwargs.pop("dataset_type")

		if dataset_type in cls.BUILDER_CLASS_TYPE:

			if cls.dataset_builders[dataset_type] is None:
				cls.dataset_builders[dataset_type] = cls.BUILDER_CLASS_TYPE[dataset_type]()

			builder = cls.dataset_builders[dataset_type]
			builder.configure(**kwargs)

			return builder.build()

		elif dataset_type in cls.DATASET_CLASS_TYPE:

			return cls.DATASET_CLASS_TYPE[dataset_type](**kwargs)

		else:

			print()
			print(f"dataset_type type: {dataset_type} is not defined")
			sys.exit()



					


