# -*- coding: utf-8 -*-
# --------------------------------- #
#       Author: Anemily Machina     #
#       Python Version: 3.10       #
#       Created on: 2023-03-15      #
# --------------------------------- #




class LabelTransformFunctions():
	
	def label_transform_dict(*args, **kwargs):

		label_dict = kwargs.pop("label_dict")
		label_index = kwargs.pop("label_index")

		if isinstance(label_index, list):

			def label_transform_fn(example):

				return [label_dict[example[l_i]] for l_i in label_index]

		else:

			def label_transform_fn(example):

				return label_dict[example[label_index]]

		return label_transform_fn


	LABEL_TRANSFORM_FN_DICT = {
		"dict":label_transform_dict
	}
	

	def get_fn(*args, **kwargs):

		label_tranform_type = kwargs.pop('label_transform_type')
		label_transform_fn = LabelTransformFunctions.LABEL_TRANSFORM_FN_DICT[label_tranform_type](**kwargs)

		return label_transform_fn

	
