# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10.4       #
#       Created on: 2022-01-31      #
# --------------------------------- #

import sys

from .context import definition


from models.builders.FFN_builder import FFNBuilder
from models.builders.granular_builder import GranularModelBuilder
from models.builders.LSTM_builder import LSTMBuilder
from models.builders.optimizer_builder import OptimizerBuilder
from models.builders.sentence_aggregation_builder import SentenceAggregationBuilder
from models.builders.sentence_pair_combination_builder import SentencePairCombinationBuilder
from models.builders.singleclass_classification_builder import SingleClassClassificationBuilder
from models.builders.word2vec_builder import Word2VecBuilder

from models.definition.flattensor2paddedtensor import FlatTensor2PaddedTensor
from models.definition.split_sentence_pairs import SplitSentencePairs




class ModuleFactory():

	BUILDER_CLASS_TYPE = {
		"word2vec":Word2VecBuilder,
		"ffn":FFNBuilder,
		"sentence_aggregation":SentenceAggregationBuilder,
		"sentence_pair_combination":SentencePairCombinationBuilder,
		"granular_model":GranularModelBuilder,
		"singleclass_model":SingleClassClassificationBuilder,
		"optimizer":OptimizerBuilder,
		"lstm":LSTMBuilder
		}

	MODULE_CLASS_TYPE = {
		"flat2pad":FlatTensor2PaddedTensor,
		"split_sentence_pairs":SplitSentencePairs
		}

	model_builders = {module_type:None for module_type in BUILDER_CLASS_TYPE}


	@classmethod
	def BUILD_MODULE(cls, *args, **kwargs):

		module_type = kwargs.pop("module_type")

		if module_type in cls.BUILDER_CLASS_TYPE:

			if cls.model_builders[module_type] is None:
				cls.model_builders[module_type] = cls.BUILDER_CLASS_TYPE[module_type]()

			builder = cls.model_builders[module_type]
			builder.configure(**kwargs)

			return builder.build()

		elif module_type in cls.MODULE_CLASS_TYPE:

			return cls.MODULE_CLASS_TYPE[module_type](**kwargs)

		else:

			print()
			print("module type: {module_type} is not defined")
			sys.exit()



					


