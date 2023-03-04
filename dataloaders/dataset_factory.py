# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

import sys

from .dataset_builders import SentencePairDatasetBuilder


class DatasetFactory():

	BUILDER_CLASS_TYPE = {
		"snli_dataset":SentencePairDatasetBuilder
		
	}

	MODULE_CLASS_TYPE = {
	
	}

	model_builders = {module_type:None for module_type in BUILDER_CLASS_TYPE}


	@classmethod
	def BUILD_DATASET(cls, *args, **kwargs):

		dataset_type = kwargs.pop("dataset_type")

		if dataset_type in cls.BUILDER_CLASS_TYPE:

			if cls.model_builders[dataset_type] is None:
				cls.model_builders[dataset_type] = cls.BUILDER_CLASS_TYPE[dataset_type]()

			builder = cls.model_builders[dataset_type]
			builder.configure(**kwargs)

			return builder.build()

		elif dataset_type in cls.MODULE_CLASS_TYPE:

			return cls.MODULE_CLASS_TYPE[dataset_type](**kwargs)

		else:

			print()
			print("dataset_type type: {module_type} is not defined")
			sys.exit()



					


