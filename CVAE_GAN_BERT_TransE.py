# -*- coding: utf-8 -*-
# ======================================================
#  Title:  VAE+GAN+BERT_Block for Knowledge Base Completion
#  Description: VAE and GAN is trained as a whole network
#               We use the pre-train embedding from TransE
#               The Encoder and Decoder are multi-channel 5 layers BERT block, we have 3 inputs, h_em, t_em and r_em
#               Early terminated condition is added
#               Dense Sampling is employed to generate triples.
#               We select the 5 closest neighbours of generated embedding,and use Discriminator to evaluate them
#               The code is based on TensorFlow and Keras, compiled with Python 3.6
#				transformer.py we used in this project is from https://github.com/Lsdefine/attention-is-all-you-need-keras
#  Author: Hao Chen
#  Date:   15th Sep 2019
# ======================================================
import sys

# reload(sys)
# sys.setdefaultencoding("utf-8")

import os
from gensim.models.keyedvectors import KeyedVectors

import numpy as np

import matplotlib as mpl

mpl.use('Agg')  # for linux server without GUI
import matplotlib.pyplot as plt
import datetime

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Embedding, Layer, concatenate, LSTM, Bidirectional
from keras.layers.core import Reshape
from keras.models import Model, Sequential, load_model
from keras import backend as K
from keras import metrics
from keras.utils import np_utils
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
# from keras.layers.advanced_activations import LeakyReLU,ELU
from keras.utils import plot_model
from keras.optimizers import Adam

from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.neighbors import NearestNeighbors
from keras.initializers import Constant
from keras.initializers import RandomNormal, Ones

from transformer import Transformer, LRSchedulerPerStep

from collections import Counter

import tensorflow as tf
import keras.backend.tensorflow_backend
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)

#data set
dataset_flag = "FB15K"
# dataset_flag = "WN18"

# process control
training_flag = True  # true: training parameters of NN model;
# false: load pretrained parameters of NN model

load_weights = False  # true: we can load pre-training weight of network
# false: we train network from very beginning

drawing_picture = False  # true: generating embedding sapce and latent space picture
# false: no pictures in generating process

# hyperparameters
original_dim = 2
embedding_dim = 100
intermediate_dim = 256
latent_dim = 64
relationship_dim = 100  # may be sparse, some realtion may be not in the data set

act_function = 'elu'
d_act_function = 'tanh'

# vae_epochs = 200
epochs = 800000
NoProgress_Iteration4EarlyTerminate = 5000
Discriminator_Start_epoch = 20000
batch_size = 64
embedding_scale = 10.  # the pre-trained embeddings are from [-10,10]

kde_sample_num = 30  # the number of sampling for each relationship

sample_ratio_gross = 0.02  # sample 10% samples of each relationship
sample_ratio_fine = 1  # keep 30% samples from the selected samples of each relationship

result_files_number = 10  # the result file number the model need to geenrate

hp_lambda = K.variable(0)
d_lambda = K.variable(0)

need_to_be_filtered_relations_fr1 = [160, 102, 176, 190, 236, 246, 278, 356, 42, 106,
									 152, 209, 378, 34, 267, 119, 28, 294, 23, 110,
									 114, 154, 101, 317, 55, 141, 177, 280, 310, 57,
									 170, 81, 121, 233, 240, 296, 199, 75, 142, 6,
									 62, 71, 15, 40, 297]

need_to_be_filtered_relations = [160, 102, 176, 190, 236, 246, 278, 356, 42, 106,
								 152, 209, 378, 34, 267, 119, 28, 294, 23, 110,
								 114, 154, 101, 317, 55, 141, 177, 280, 310, 57,
								 170, 81, 121, 233, 240, 296, 199, 75, 142, 6,
								 62, 71, 15, 40, 297, 395, 242, 82, 178, 217,
								 446, 133, 438, 315, 324, 98, 109, 173]

# need_to_be_filtered_relations = []

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


# slice e1_e2 into e1 and e2
def SliceTensor(x, index):
	return x[:, index, :]


# generate gensim KeyedVectors by myself,just want to use most_similar function
def GenerateEmbeddingKeyedVectors(embedding_narray):
	KeyedVector = KeyedVectors(embedding_narray.shape[1])
	KeyedVector.vectors = embedding_narray

	nIndex2Word = np.arange(embedding_narray.shape[0])
	list = []
	for item in nIndex2Word:
		list.append(str(item))
	Index2Word = np.array(list)
	KeyedVector.index2entity = Index2Word
	KeyedVector.index2word = Index2Word

	return KeyedVector


# Just use cartesian to get all the pairs of head entity list and tail entity list
def cartesian(arrays, out=None):
	"""
	Generate a cartesian product of input arrays.

	Parameters
	----------
	arrays : list of array-like
		1-D arrays to form the cartesian product of.
	out : ndarray
		Array to place the cartesian product in.

	Returns
	-------
	out : ndarray
		2-D array of shape (M, len(arrays)) containing cartesian products
		formed of input arrays.

	Examples
	--------
	>>cartesian(([1, 2, 3], [4, 5], [6, 7]))
	array([[1, 4, 6],
		   [1, 4, 7],
		   [1, 5, 6],
		   [1, 5, 7],
		   [2, 4, 6],
		   [2, 4, 7],
		   [2, 5, 6],
		   [2, 5, 7],
		   [3, 4, 6],
		   [3, 4, 7],
		   [3, 5, 6],
		   [3, 5, 7]])

	"""

	arrays = [np.asarray(x) for x in arrays]

	dtype = arrays[0].dtype
	# find max dtype, the aviod cut error
	for item in arrays:
		if item.dtype > dtype:
			dtype = item.dtype

	n = np.prod([x.size for x in arrays])
	if out is None:
		out = np.zeros([n, len(arrays)], dtype=dtype)

	m = n // arrays[0].size
	out[:, 0] = np.repeat(arrays[0], m)
	if arrays[1:]:
		cartesian(arrays[1:], out=out[0:m, 1:])
		for j in range(1, arrays[0].size):
			out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
	return out


# shuffle two arrays in unison way
def shuffle_in_unison(a, b):
	assert len(a) == len(b)
	shuffled_a = np.empty(a.shape, dtype=a.dtype)
	shuffled_b = np.empty(b.shape, dtype=b.dtype)
	permutation = np.random.permutation(len(a))
	for old_index, new_index in enumerate(permutation):
		shuffled_a[new_index] = a[old_index]
		shuffled_b[new_index] = b[old_index]
	return shuffled_a, shuffled_b


# get the sampling index
class BatchSampling():
	def __init__(self, total_size):
		self.nStartIndex = 0
		self.nTotalSize = total_size

	# get the index randomly
	def RandomSampling_index(self, s_batch_size):
		idx = np.random.randint(0, self.nTotalSize, s_batch_size)
		return idx

	# get the index block by block
	def BlockSampling_index(self, s_batch_size):
		shuffleFlag = False  # if the dataset should be shuffled
		if self.nStartIndex + s_batch_size < self.nTotalSize:
			nEndIndex = self.nStartIndex + s_batch_size
			idx = np.arange(self.nStartIndex, nEndIndex)
			self.nStartIndex = nEndIndex
		else:
			idx_1 = np.arange(self.nStartIndex, self.nTotalSize)
			idx_2 = np.random.randint(0, self.nStartIndex, self.nStartIndex + s_batch_size - self.nTotalSize)
			idx = np.concatenate([idx_1, idx_2])
			self.nStartIndex = 0
			shuffleFlag = True

		return idx, shuffleFlag


class VAE_GAN():
	def __init__(self, pre_train_entity_embedding_file_name, pre_train_relation_embedding_file_name):
		self.PreTrainEmbeddingFileName = pre_train_entity_embedding_file_name

		# self.EntityVectors = self.LoadEmbaddingFile()

		# load entity and relationship embedding form numpy array
		# entity_embedding_narray = np.load("./data/FB15K_TransE/entity_embeddings.npy")
		# relation_embedding_narray = np.load("./data/FB15K_TransE/relation_embeddings.npy")

		entity_embedding_narray = np.load(pre_train_entity_embedding_file_name)
		relation_embedding_narray = np.load(pre_train_relation_embedding_file_name)

		self.EntityVectors = GenerateEmbeddingKeyedVectors(entity_embedding_narray)
		self.RelationVectors = GenerateEmbeddingKeyedVectors(relation_embedding_narray)

		self.EntityEmbeddingLayer = self.BuildEmbedidngLayer(self.EntityVectors)

	# Load the Pretrained Embedding File
	# output: the mapping dictionary from Entity to Vectors
	# def LoadEmbaddingFile(self):
	#     EntityVectors = KeyedVectors.load_word2vec_format(self.PreTrainEmbeddingFileName, binary=True)
	#     return EntityVectors

	# Build the Embedidng Layer from the pre-trained embedding vectors
	# input: Pre-trained EntityVectors
	# output: embedding layer modle
	def BuildEmbedidngLayer(self, EntityVectors):
		weights = EntityVectors.vectors
		# entity2vec_embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
		#                   output_dim=embedding_matrix.shape[1],
		#                   embeddings_initializer=Constant(embedding_matrix),
		#                   trainable=False)

		# entity2vec_embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
		#                                        output_dim=embedding_matrix.shape[1],
		#                                        embeddings_initializer=RandomNormal(mean=0.0, stddev=1, seed=None),
		#                                        # embeddings_initializer=Ones(),
		#                                        trainable=True)

		entity2vec_embedding_layer = Embedding(input_dim=weights.shape[0],
											   output_dim=weights.shape[1],
											   weights=[weights],
											   trainable=False)  # if we can train the embedding, will the result be better?

		x = Input((original_dim,))
		embed = entity2vec_embedding_layer(x)
		embedding = Model(x, embed)

		return embedding

	# instead of sampling from Q(z|X), sample eps = N(0,I)
	# reparameterization trick: z = z_mean + sqrt(var)*eps
	def sampling(self, args):
		"""Reparameterization trick by sampling fr an isotropic unit Gaussian.
		# Arguments:
			args (tensor): mean and log of variance of Q(z|X)
		# Returns:
			z (tensor): sampled latent vector
		"""

		z_mean, z_log_var = args
		batch = K.shape(z_mean)[0]
		dim = K.int_shape(z_mean)[1]
		# by default, random_normal has mean=0 and std=1.0
		epsilon = K.random_normal(shape=(batch, dim))
		return z_mean + K.exp(0.5 * z_log_var) * epsilon

	# genetate the encoder,inputs shape(2,1000), output shape(200, 200, 200)
	def BuildEncoder(self):
		x_emb = Input((2, embedding_dim,))

		y = Input((relationship_dim,))
		r_em = Reshape((1, relationship_dim))(y)   #expand 1D for y to use multi-head attention (batch, 100)->(batch,1,100)

		enc_input = concatenate([x_emb, r_em], axis=1)

		# layer1
		transformer_layer_11 = Transformer(len_limit=3, d_model=100, d_inner_hid=512,\
										n_head=5, layers=1, dropout=0.1)
		transformer_layer_12 = Transformer(len_limit=3, d_model=100, d_inner_hid=512,\
										 n_head=5, layers=1, dropout=0.1)
		transformer_layer_13 = Transformer(len_limit=3, d_model=100, d_inner_hid=512,\
										n_head=5, layers=1, dropout=0.1)

		h_out1s = transformer_layer_11.generate_Encoder_layer(enc_input)
		t_out1s = transformer_layer_12.generate_Encoder_layer(enc_input)
		r_out1s = transformer_layer_13.generate_Encoder_layer(enc_input)

		h_out1 = Lambda(SliceTensor, arguments={'index': 0})(h_out1s)
		t_out1 = Lambda(SliceTensor, arguments={'index': 1})(t_out1s)
		r_out1 = Lambda(SliceTensor, arguments={'index': 2})(r_out1s)

		h_out1 = Reshape((1, relationship_dim))(h_out1)
		t_out1 = Reshape((1, relationship_dim))(t_out1)
		r_out1 = Reshape((1, relationship_dim))(r_out1)
		ly1_result = concatenate([h_out1, t_out1, r_out1], axis=1)

		# layer2
		h_out2s = transformer_layer_11.generate_Encoder_layer(ly1_result)
		t_out2s = transformer_layer_12.generate_Encoder_layer(ly1_result)
		r_out2s = transformer_layer_13.generate_Encoder_layer(ly1_result)

		h_out2 = Lambda(SliceTensor, arguments={'index': 0})(h_out2s)
		t_out2 = Lambda(SliceTensor, arguments={'index': 1})(t_out2s)
		r_out2 = Lambda(SliceTensor, arguments={'index': 2})(r_out2s)
		h_out2 = Reshape((1, relationship_dim))(h_out2)
		t_out2 = Reshape((1, relationship_dim))(t_out2)
		r_out2 = Reshape((1, relationship_dim))(r_out2)

		ly2_result = concatenate([h_out2, t_out2, r_out2], axis=1)

		transformer_layer_output = Transformer(len_limit=4, d_model=100, d_inner_hid=512, \
										   n_head=5, layers=2, dropout=0.1)

		enc_outputs = transformer_layer_output.generate_Encoder_layer(ly2_result)

		enc_output = Reshape((3 * 100,))(enc_outputs)  #(batch, 9, 100)->(batch,900)

		h_dense1 = Dense(1024, activation=act_function, name='h_dense1')(enc_output)
		h_dense2 = Dense(256, activation=act_function, name='h_dense2')(h_dense1)
		h_ntl = Dense(64, activation=act_function, name='h_dense3')(h_dense2)


		# h_dense6_1 = Reshape((2, 128))(h_dense6)
		# e_1 = Lambda(SliceTensor, arguments={'index': 0})(h_dense6_1)
		# e_2 = Lambda(SliceTensor, arguments={'index': 1})(h_dense6_1)
		#
		# h_ntl = NeuralTensorLayer(output_dim=64, input_dim=128)([e_1, e_2])

		# h_ntl_1 = Lambda(SliceTensor, arguments={'index': 0})(h_ntl)
		h_cond = concatenate([h_ntl, y])


		# h_cond_2D = concatenate([h_ntl, y_2D])

		# h_cond = Lambda(SliceTensor, arguments={'index': 0})(h_cond_2D)

		z_mean = Dense(latent_dim, name='z_mean')(h_cond)
		z_log_var = Dense(latent_dim, name='z_log_var')(h_cond)

		# use reparameterization trick to push the sampling out as input
		# note that "output_shape" isn't necessary with the TensorFlow backend
		z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
		# y = Lambda(SliceTensor, arguments={'index': 0})(y_2D)

		z_cond = concatenate([z, y])
		# z_cond = z

		encoder = Model([x_emb, y], [z_mean, z_log_var, z_cond], name='encoder')

		# visiualize the neural network
		encoder.summary()
		# plot_model(encoder, to_file='VAE_GAN_encoder.png', show_shapes=True)
		return encoder

	def BuildEncoder1(self):
		x = Input((2, embedding_dim,))
		y = Input((relationship_dim,))

		h_t = Dense(intermediate_dim, activation=act_function, name='h')(x)
		h_merge = Reshape((2 * intermediate_dim,))(h_t)  # shape(2,1000)->shape(2000)
		# h_merge = h_t
		h_dense = Dense(1792, activation=act_function, name='h_dense')(h_merge)
		h_dense2 = Dense(1000, activation=act_function, name='h_dense2')(h_dense)
		h_dense3 = Dense(640, activation=act_function, name='h_dense3')(h_dense2)
		h_dense4 = Dense(640, activation=act_function, name='h_dense4')(h_dense3)
		h_dense5 = Dense(512, activation=act_function, name='h_dense5')(h_dense4)
		h_dense6 = Dense(256, activation=act_function, name='h_dense6')(h_dense5)

		h_dense6_1 = Reshape((2, 128))(h_dense6)
		e_1 = Lambda(SliceTensor, arguments={'index': 0})(h_dense6_1)
		e_2 = Lambda(SliceTensor, arguments={'index': 1})(h_dense6_1)

		# h_ntl = NeuralTensorLayer(output_dim=32, input_dim=64)([h_dense6, h_dense6])
		h_ntl = NeuralTensorLayer(output_dim=64, input_dim=128)([e_1, e_2])

		h_cond = concatenate([h_ntl, y])

		z_mean = Dense(latent_dim, name='z_mean')(h_cond)
		z_log_var = Dense(latent_dim, name='z_log_var')(h_cond)

		# use reparameterization trick to push the sampling out as input
		# note that "output_shape" isn't necessary with the TensorFlow backend
		z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
		z_cond = concatenate([z, y])

		encoder = Model([x, y], [z_mean, z_log_var, z_cond], name='encoder')

		# visiualize the neural network
		encoder.summary()
		# plot_model(encoder, to_file='VAE_GAN_encoder.png', show_shapes=True)
		return encoder

	def BuildDecoder(self):

		latent_inputs_ora = Input(shape=(latent_dim + relationship_dim,), name='z_sampling')  # sampling z and add r
		latent_inputs_medium = Reshape((1, latent_dim + relationship_dim))(latent_inputs_ora)  # expand 1D for latent_inputs to use multi-head attention (batch, 164)->(batch,1,164)
		latent_inputs = concatenate([latent_inputs_medium, latent_inputs_medium, latent_inputs_medium], axis=1)  # expand 1D for latent_inputs to use multi-head attention (batch, 164)->(batch,1,164)

		#layer1
		transformer_layer_11 = Transformer(len_limit=7, d_model=latent_dim + relationship_dim, d_inner_hid=512,
										n_head=4, layers=1, dropout=0.1)
		transformer_layer_12 = Transformer(len_limit=7, d_model=latent_dim + relationship_dim, d_inner_hid=512,
										n_head=4, layers=1, dropout=0.1)
		transformer_layer_13 = Transformer(len_limit=7, d_model=latent_dim + relationship_dim, d_inner_hid=512,
										n_head=4, layers=1, dropout=0.1)

		zr_out11s = transformer_layer_11.generate_Encoder_layer(latent_inputs)
		zr_out12s = transformer_layer_12.generate_Encoder_layer(latent_inputs)
		zr_out13s = transformer_layer_13.generate_Encoder_layer(latent_inputs)

		zr_out11 = Lambda(SliceTensor, arguments={'index': 0})(zr_out11s)
		zr_out12 = Lambda(SliceTensor, arguments={'index': 1})(zr_out12s)
		zr_out13 = Lambda(SliceTensor, arguments={'index': 2})(zr_out13s)
		zr_out11 = Reshape((1, (latent_dim + relationship_dim)))(zr_out11)
		zr_out12 = Reshape((1, (latent_dim + relationship_dim)))(zr_out12)
		zr_out13 = Reshape((1, (latent_dim + relationship_dim)))(zr_out13)

		ly1_result = concatenate([zr_out11, zr_out12, zr_out13], axis=1)

		# layer2
		zr_out21s = transformer_layer_11.generate_Encoder_layer(ly1_result)
		zr_out22s = transformer_layer_12.generate_Encoder_layer(ly1_result)
		zr_out23s = transformer_layer_13.generate_Encoder_layer(ly1_result)

		zr_out21 = Lambda(SliceTensor, arguments={'index': 0})(zr_out21s)
		zr_out22 = Lambda(SliceTensor, arguments={'index': 1})(zr_out22s)
		zr_out23 = Lambda(SliceTensor, arguments={'index': 2})(zr_out23s)
		zr_out21 = Reshape((1, (latent_dim + relationship_dim)))(zr_out21)
		zr_out22 = Reshape((1, (latent_dim + relationship_dim)))(zr_out22)
		zr_out23 = Reshape((1, (latent_dim + relationship_dim)))(zr_out23)

		ly2_result = concatenate([zr_out21, zr_out22, zr_out23], axis=1)


		transformer_layer_output = Transformer(len_limit=7, d_model=latent_dim + relationship_dim, d_inner_hid=512,
										n_head=4, layers=2, dropout=0.1)

		trsf_outputs = transformer_layer_output.generate_Encoder_layer(ly2_result)

		trsf_output = Reshape((3 * (latent_dim + relationship_dim),))(trsf_outputs)

		decoder_h5 = Dense(1000, activation=act_function)(trsf_output)
		# decoder_h6 = Lambda(SliceTensor, arguments={'index': 0})(decoder_h5)

		h_merge = Dense(2 * intermediate_dim, name='z_mean')(decoder_h5)
		h_t = Reshape((2, intermediate_dim))(h_merge)  # shape(2000)->shape(2,1000)
		ht_embed = Dense(embedding_dim, activation='linear')(h_t)

		decoder = Model(latent_inputs_ora, ht_embed, name='generator')
		# visiualize the neural network
		decoder.summary()
		# plot_model(decoder, to_file='VAE_GAN_decoder.png', show_shapes=True)
		return decoder

	def BuildDecoder1(self):
		latent_inputs = Input(shape=(latent_dim + relationship_dim,), name='z_sampling')  # sampling z and add r

		decoder_h = Dense(256, activation=act_function)(latent_inputs)
		decoder_h2 = Dense(512, activation=act_function)(decoder_h)
		decoder_h3 = Dense(640, activation=act_function)(decoder_h2)
		decoder_h4 = Dense(640, activation=act_function)(decoder_h3)
		decoder_h5 = Dense(1000, activation=act_function)(decoder_h4)
		decoder_h6 = Dense(1792, activation=act_function)(decoder_h5)
		h_merge = Dense(2 * intermediate_dim, name='z_mean')(decoder_h6)
		h_t = Reshape((2, intermediate_dim))(h_merge)  # shape(2000)->shape(2,1000)
		ht_embed = Dense(embedding_dim, activation='linear')(h_t)

		decoder = Model(latent_inputs, ht_embed, name='generator')

		# visiualize the neural network
		decoder.summary()
		# plot_model(decoder, to_file='VAE_GAN_decoder.png', show_shapes=True)
		return decoder

	# build the discriminator for GAN
	# we have two inputs, namely: [embedding_h, embedding_t], relation_one_hot
	def BuildDisctriminator(self):
		x = Input((2, embedding_dim,))

		y = Input((relationship_dim,))

		d_merge = Reshape((2 * embedding_dim,))(x)  # shape(2,1000)->shape(2000)
		d_htr = concatenate([d_merge, y])
		# d0 = Dense(1700, activation=act_function, name='d0')(d_htr)
		d1 = Dense(1024, activation=d_act_function, name='d1')(d_htr)
		# d2 = Dense(512, activation=d_act_function, name='d2')(d1)
		d3 = Dense(256, activation=d_act_function, name='d3')(d1)
		d4 = Dense(64, activation=d_act_function, name='d4')(d3)
		d_result = Dense(1, activation='sigmoid', name='d_result')(d4)

		discriminator = Model([x, y], d_result, name='discriminator')

		# visiualize the neural network
		discriminator.summary()
		# plot_model(discriminator, to_file='VAE_GAN_discriminator.png', show_shapes=True)
		return discriminator


def train(self):
	print("training process is done")
	return


def EvaluateModel(self):
	return


# Custom loss layer, define customized loss function
class CustomVariationalLayer(Layer):
	def __init__(self, **kwargs):
		self.is_placeholder = True
		super(CustomVariationalLayer, self).__init__(**kwargs)

	def vae_loss(self, embed_input, x_decoded_mean):
		reconstruction_loss = original_dim * embedding_dim * metrics.mean_squared_error(
			K.reshape(embed_input, (-1, original_dim * embedding_dim)),
			K.reshape(x_decoded_mean, (-1, original_dim * embedding_dim)))

		kl_loss = - 0.5 * K.sum(1. + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

		return K.mean(reconstruction_loss + hp_lambda * kl_loss)

	def call(self, inputs):
		embed_input = inputs[0]
		x_decoded_mean = inputs[1]
		loss = self.vae_loss(embed_input, x_decoded_mean)
		self.add_loss(loss, inputs=inputs)
		# We won't actually use the output.
		return embed_input


class CustomCombinedNNLossLayer(Layer):
	def __init__(self, **kwargs):
		self.is_placeholder = True
		super(CustomCombinedNNLossLayer, self).__init__(**kwargs)

	def combinedNN_loss(self, embed_input, x_decoded_mean, validate_lable, validate_result):
		reconstruction_loss = original_dim * embedding_dim * metrics.mean_squared_error(  # binary_crossentropy
			K.reshape(embed_input, (-1, original_dim * embedding_dim)),
			K.reshape(x_decoded_mean, (-1, original_dim * embedding_dim)))
		kl_loss = - 0.5 * K.sum(1. + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

		discriminator_loss = metrics.binary_crossentropy(validate_lable, validate_result)
		return K.mean(reconstruction_loss + hp_lambda * kl_loss + d_lambda * discriminator_loss)

	def call(self, inputs):
		embed_input = inputs[0]
		x_decoded_mean = inputs[1]
		validate_lable = inputs[2]
		validate_result = inputs[3]
		loss = self.combinedNN_loss(embed_input, x_decoded_mean, validate_lable, validate_result)
		self.add_loss(loss, inputs=inputs)
		# We won't actually use the output.
		return embed_input


class NeuralTensorLayer(Layer):
	def __init__(self, output_dim, input_dim=None, **kwargs):
		self.output_dim = output_dim  # k
		self.input_dim = input_dim  # d
		if self.input_dim:
			kwargs['input_shape'] = (self.input_dim,)
		super(NeuralTensorLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		mean = 0.0
		std = 1.0
		# W : k*d*d
		k = self.output_dim
		d = self.input_dim
		initial_W_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(k, d, d))
		initial_V_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(2 * d, k))
		self.W = K.variable(initial_W_values)
		self.V = K.variable(initial_V_values)
		self.b = K.zeros((self.input_dim,))
		self.trainable_weights = [self.W, self.V, self.b]

	def call(self, inputs, mask=None):
		if type(inputs) is not list or len(inputs) <= 1:
			raise Exception('BilinearTensorLayer must be called on a list of tensors '
							'(at least 2). Got: ' + str(inputs))
		e1 = inputs[0]
		e2 = inputs[1]
		batch_size = K.shape(e1)[0]
		k = self.output_dim
		# print([e1,e2])
		feed_forward_product = K.dot(K.concatenate([e1, e2]), self.V)
		# print(feed_forward_product)
		bilinear_tensor_products = [K.sum((e2 * K.dot(e1, self.W[0])) + self.b, axis=1)]
		# print(bilinear_tensor_products)
		for i in range(k)[1:]:
			btp = K.sum((e2 * K.dot(e1, self.W[i])) + self.b, axis=1)
			bilinear_tensor_products.append(btp)
		result = K.tanh(
			K.reshape(K.concatenate(bilinear_tensor_products, axis=0), (batch_size, k)) + feed_forward_product)
		# print(result)
		return result

	def compute_output_shape(self, input_shape):
		# print (input_shape)
		batch_size = input_shape[0][0]
		return (batch_size, self.output_dim)


# if active=true, do normalize or denormalize,else do nothing
def embedding_rescale(numpy_matrix, active=False, scale=embedding_scale, inverse=False):
	if active == True:
		if inverse == True:
			return (numpy_matrix * 2 * scale) - scale
		else:
			return (numpy_matrix + scale) / (2 * scale)
	else:
		return numpy_matrix


# read all the data(including training, validation, test) from data file
# input: data file list: the sequence is training, validation, test
# output: the 2-dim list of the data
def Load_all_data(data_file_name_list):
	WholeH_TEntityIDList = []
	WholeRelationList = []
	for DataFileName in data_file_name_list:
		H_TEntityIDList, RelationList = Load_each_data_file(DataFileName)
		WholeH_TEntityIDList.append(H_TEntityIDList)
		WholeRelationList.append(RelationList)

	return WholeH_TEntityIDList, WholeRelationList


# read head entity, tail entity and relation form datafile
# input: data file name
# output: the list of entity pairs and relations
def Load_each_data_file(data_file_name):
	DataFile = open(data_file_name, "r")
	TotalLineNum = (int)(DataFile.readline())

	H_TEntityIDList = []
	RelationList = []
	for i in range(TotalLineNum):
		content = DataFile.readline()
		h, t, r = content.strip().split()

		# if int(r) != 242:       #only load relationship 242
		#     continue

		EntityPair = [int(h), int(t)]  # string to int
		H_TEntityIDList.append(EntityPair)
		RelationList.append(int(r))  # string to int

	return H_TEntityIDList, RelationList


# sampling form z-space and generate the input of decoder
# input:  batch:batchsize; lat_dim: latent dimension; r_dim: relation dimention(one-hot)
# output: the tensor generated randomly
def SampleFormSpaceZ(batch, lat_dim, np_Y):
	# random genarate h_t in latent space
	r_ht = np.random.normal(0, 1, (batch, lat_dim))

	# random generate r as one_hot_vactor
	# r_relation = np.random.randint(0, r_dim, batch)
	# r_relation_onehot = np_utils.to_categorical(r_relation, num_classes=r_dim)  # to one-hot vector

	id_r = np.random.randint(0, np_Y.shape[0], batch)
	r_relation = np_Y[id_r]

	z_r_vector = np.hstack(
		(r_ht, r_relation))  # concatenate the r_ht and r_relation_onehot(the input of decoder)

	return z_r_vector, r_relation


# display a 2D plot of the digit classes in the latent space
def VisulaizelatentSpace(np_X, np_Y, Y_label, batch_size, file_path_name):
	X_encoded_all = encoder.predict([np_X, np_Y], batch_size=batch_size)
	X_encoded = X_encoded_all[0]
	model_pca = PCA(n_components=2)
	X_encoded_pca = model_pca.fit_transform(X_encoded)
	plt.figure(figsize=(6, 6))
	unique = list(set(Y_label))
	# http://blog.olgabotvinnik.com/blog/2013/08/21/2013-08-21-prettyplotlib-painlessly-create-beautiful-matplotlib/
	colors = np.arange(relationship_dim)
	for i, u in enumerate(unique):
		xi = [X_encoded_pca[j, 0] for j in range(X_encoded_pca.shape[0]) if Y_label[j] == u]
		yi = [X_encoded_pca[j, 1] for j in range(X_encoded_pca.shape[0]) if Y_label[j] == u]
		plt.scatter(xi, yi, label=str(u), marker="o", alpha=.5, s=8)
		# plt.scatter(xi, yi, label=str(u), marker="o", alpha=.5)
		# plt.scatter(xi, yi, label=str(u).replace('2', r'$\to$'), marker="o", alpha=.5, s=8)

	lgnd = plt.legend(loc='upper right', ncol=2)
	# for i in np.arange(relationship_dim):
	#     lgnd.legendHandles[i]._sizes = [30]
	plt.savefig(file_path_name, bbox_inches='tight')
	# plt.show()


# display a 2D plot of the digit classes in the latent space
# if the relationship is more than 10, we will generate multi-graphs
def VisulaizelatentSpace4LargeData(np_X, np_Y, Y_lable, batch_size, file_path_name):
	X_encoded_all = encoder.predict([np_X, np_Y], batch_size=batch_size)
	X_encoded = X_encoded_all[0]
	model_pca = PCA(n_components=2)
	X_encoded_pca = model_pca.fit_transform(X_encoded)

	unique = list(set(Y_lable))
	# http://blog.olgabotvinnik.com/blog/2013/08/21/2013-08-21-prettyplotlib-painlessly-create-beautiful-matplotlib/
	colors = np.arange(relationship_dim)
	plt.figure(figsize=(6, 6))
	picture_index = 0
	for i, u in enumerate(unique):
		xi = [X_encoded_pca[j, 0] for j in range(X_encoded_pca.shape[0]) if Y_lable[j] == u]
		yi = [X_encoded_pca[j, 1] for j in range(X_encoded_pca.shape[0]) if Y_lable[j] == u]
		plt.scatter(xi, yi, label=str(u), marker="o", alpha=.5, s=8)

		if (i % 10 == 0 and i != 0):  # how many relationships in one picture
			lgnd = plt.legend(loc='upper right', ncol=2)
			plt.savefig(file_path_name + "_" + str(picture_index) + ".pdf", bbox_inches='tight')

			plt.figure(figsize=(6, 6))
			picture_index = picture_index + 1

	# for i in np.arange(relationship_dim):
	#     lgnd.legendHandles[i]._sizes = [30]
	lgnd = plt.legend(loc='upper right', ncol=2)
	plt.savefig(file_path_name + "_" + str(picture_index) + ".pdf", bbox_inches='tight')
	# plt.show()


# display a 2D plot of the digit classes in the latent space
def VisulaizelatentSpace_withsamplingdata(np_X, Y_onehot, np_Y, batch_size, file_path_name, zsample_data):
	X_encoded_all = encoder.predict([np_X, Y_onehot], batch_size=batch_size)
	X_encoded = X_encoded_all[0]
	model_pca = PCA(n_components=2)
	X_encoded_pca = model_pca.fit_transform(X_encoded)
	plt.figure(figsize=(6, 6))
	unique = list(set(np_Y))
	# http://blog.olgabotvinnik.com/blog/2013/08/21/2013-08-21-prettyplotlib-painlessly-create-beautiful-matplotlib/
	colors = np.arange(relationship_dim)
	for i, u in enumerate(unique):
		xi = [X_encoded_pca[j, 0] for j in range(X_encoded_pca.shape[0]) if np_Y[j] == u]
		yi = [X_encoded_pca[j, 1] for j in range(X_encoded_pca.shape[0]) if np_Y[j] == u]
		plt.scatter(xi, yi, label=str(u), marker="o", alpha=.5, s=8)
		# plt.scatter(xi, yi, label=str(u), marker="o", alpha=.5)
		# plt.scatter(xi, yi, label=str(u).replace('2', r'$\to$'), marker="o", alpha=.5, s=8)

	zsample_data_pca = model_pca.fit_transform(zsample_data)
	xi = zsample_data_pca[:, 0]
	yi = zsample_data_pca[:, 1]
	plt.scatter(xi, yi, label="sample", marker="o", alpha=.5, s=8)

	lgnd = plt.legend(loc='upper right', ncol=2)
	# for i in np.arange(relationship_dim):
	#     lgnd.legendHandles[i]._sizes = [30]
	plt.savefig(file_path_name, bbox_inches='tight')
	# plt.show()


# display a 2D plot in sampling data
def VisulaizelatentSpace4sampling(X_encoded, file_path_name):
	model_pca = PCA(n_components=2)
	X_encoded_pca = model_pca.fit_transform(X_encoded)
	plt.figure(figsize=(6, 6))

	xi = X_encoded_pca[:, 0]
	yi = X_encoded_pca[:, 1]

	plt.scatter(xi, yi, marker="o", alpha=.5, s=8)
	plt.savefig(file_path_name, bbox_inches='tight')
	# plt.show()


# visualize the embedding space compare the original data with generate data
def VisulaizeEmbeddingSpace(X_ori, Y_onehot, np_Y, file_path_name):
	X_encoded_all = encoder.predict([X_ori, Y_onehot], batch_size=batch_size)
	X_encoded = X_encoded_all[0]
	X_gen = generator.predict(np.concatenate([X_encoded, Y_onehot], axis=1))

	model_pca = PCA(n_components=2)
	X_ori_pca = model_pca.fit_transform(X_ori.reshape(-1, 2 * embedding_dim))
	X_gen_pca = model_pca.fit_transform(X_gen.reshape(-1, 2 * embedding_dim))

	plt.figure(figsize=(6, 6))
	unique = list(set(np_Y))
	# http://blog.olgabotvinnik.com/blog/2013/08/21/2013-08-21-prettyplotlib-painlessly-create-beautiful-matplotlib/
	colors = np.arange(relationship_dim)
	for i, u in enumerate(unique):
		xi_1 = [X_ori_pca[j, 0] for j in range(X_ori_pca.shape[0]) if np_Y[j] == u]
		yi_1 = [X_ori_pca[j, 1] for j in range(X_ori_pca.shape[0]) if np_Y[j] == u]
		plt.scatter(xi_1, yi_1, label=str(u), marker="o", alpha=.5, s=8)

		# plt.scatter(xi, yi, label=str(u), marker="o", alpha=.5)
		# plt.scatter(xi, yi, label=str(u).replace('2', r'$\to$'), marker="o", alpha=.5, s=8)

	for i, u in enumerate(unique):
		xi_2 = [X_gen_pca[j, 0] for j in range(X_gen_pca.shape[0]) if np_Y[j] == u]
		yi_2 = [X_gen_pca[j, 1] for j in range(X_gen_pca.shape[0]) if np_Y[j] == u]
		plt.scatter(xi_2, yi_2, label=str(u) + "-g", marker="x", alpha=.5, s=8)

	lgnd = plt.legend(loc='upper right', ncol=2)
	# for i in np.arange(relationship_dim):
	#     lgnd.legendHandles[i]._sizes = [30]
	plt.savefig(file_path_name, bbox_inches='tight')


# compute the Reconstruction Loss of the data
def ComputingReconstructionLoss(np_X, Y_onehot):
	X_encoded_all = encoder.predict([np_X, Y_onehot], batch_size=batch_size * 10)
	Z_sample_mean = X_encoded_all[0]
	x_generated = generator.predict(np.concatenate([Z_sample_mean, Y_onehot], axis=1), batch_size=batch_size * 5)
	reconstuction_loss = 2 * embedding_dim * mean_squared_error(
		np_X.reshape(-1, embedding_dim * 2), x_generated.reshape(-1, embedding_dim * 2))
	return reconstuction_loss


# compute the Reconstruction Loss of the data
# since the data set may be too large, we just select some tuples by random to do the loss computation
# tuple_num is the number of tuples we selected to do the loss computation
def ComputingReconstructionLoss_RandomBatch(np_X, np_Y, tuple_num):
	ids = np.random.randint(0, np_X.shape[0], tuple_num)
	np_X_selected = np_X[ids]
	np_Y_selected = np_Y[ids]

	X_encoded_all = encoder.predict([np_X_selected, np_Y_selected], batch_size=batch_size)
	Z_sample_mean = X_encoded_all[0]
	x_generated = generator.predict(np.concatenate([Z_sample_mean, np_Y_selected], axis=1))
	reconstuction_loss = 2 * embedding_dim * mean_squared_error(
		np_X_selected.reshape(-1, embedding_dim * 2), x_generated.reshape(-1, embedding_dim * 2))
	return reconstuction_loss


# compute the Loss of the data
# since the data set may be too large, we just select some tuples by random to do the loss computation
# tuple_num is the number of tuples we selected to do the loss computation
def ComputingGenerationLoss_RandomBatch(np_X, np_Y, tuple_num):
	ids = np.random.randint(0, np_X.shape[0], tuple_num)
	np_X_selected = np_X[ids]
	np_Y_selected = np_Y[ids]
	c_valid = np.ones((tuple_num, 1))

	g_loss = combined.test_on_batch([np_X_selected, np_Y_selected, c_valid], None)
	return g_loss


# create a dictionary including in the Path
def mkdir(path):
	path = path.strip()
	path = path.rstrip("\\")

	isExists = os.path.exists(path)

	if not isExists:
		os.makedirs(path)
		return True
	else:
		print(path + ' not exist')
		return False


# using dense sampling, and sample the same number of triplets from each relation
def DenseSamplingFromZ_Snum(kde_sample_num):
	# ///////////////sample Z(latent) form Gaussian distribution(fitted form training data)/////////////
	gaussian_kde_distributions = map(
		lambda r_id: stats.gaussian_kde(np.transpose(X_train_encoded[np.where(Y_train == r_id)])),
		Y_lable_unique)

	Z_sample = np.transpose(
		np.concatenate(map(lambda kde: kde.resample(kde_sample_num), gaussian_kde_distributions), axis=1))

	Y_sample = np.concatenate(map(lambda rid: np.broadcast_to(
		KBC_model.RelationVectors.vectors[rid],
		(kde_sample_num, relationship_dim)), Y_lable_unique))

	# x_decoded = embedding_rescale(generator.predict(np.concatenate([Z_sample, Y_sample], axis=1)), embedding_scale,
	#                               inverse=True)

	x_decoded = generator.predict(np.concatenate([Z_sample, Y_sample], axis=1))

	return x_decoded, Z_sample, Y_sample


# using dense sampling, and sample the same number of triplets from each relation
def DenseSamplingFromZ_SRatio(sample_ratio, RelationFreq_list):
	Z_sample = []
	Y_sample = []
	for item in RelationFreq_list:
		r_id = item[0]
		kde_sample_num = int(item[1] * sample_ratio)

		if kde_sample_num * sample_ratio_fine < 2:  # if the number of generative samples is too small, just skip
			continue

		if r_id in need_to_be_filtered_relations:
			continue

		gaussian_kde_distribution = stats.gaussian_kde(np.transpose(X_train_encoded[np.where(Y_train == r_id)]))
		Z_sample_r = np.transpose(gaussian_kde_distribution.resample(kde_sample_num))
		Z_sample.append(Z_sample_r)

		Y_sample_r = np.broadcast_to(KBC_model.RelationVectors.vectors[r_id], (kde_sample_num, relationship_dim))
		Y_sample.append(Y_sample_r)

	Z_sample = np.concatenate(Z_sample)
	Y_sample = np.concatenate(Y_sample)

	x_decoded = generator.predict(np.concatenate([Z_sample, Y_sample], axis=1))

	return x_decoded, Z_sample, Y_sample


# x_item, y_item: the item generated by model
def EvaluateNeibor4SingleSample_Mulpy(x_item, y_item, NeighbourScope, Dtrust_threshold, similarity_threshold):
	Predict_Epair = []
	head_entityID_list = KBC_model.EntityVectors.most_similar(
		x_item[0, :].reshape(1, embedding_dim), topn=NeighbourScope)

	tail_entityID_list = KBC_model.EntityVectors.most_similar(
		x_item[1, :].reshape(1, embedding_dim), topn=NeighbourScope)

	cur_relation = KBC_model.RelationVectors.most_similar(y_item.reshape(1, embedding_dim), topn=1)[0][0]

	head_entityID_dstflt_list = []
	for item in head_entityID_list:
		if item[1] >= similarity_threshold:
			head_entityID_dstflt_list.append(item)

	tail_entityID_dstflt_list = []
	for item in tail_entityID_list:
		if item[1] >= similarity_threshold:
			tail_entityID_dstflt_list.append(item)

	if len(head_entityID_dstflt_list) == 0:
		head_entityID_dstflt_list.append(head_entityID_list[0])
	if len(tail_entityID_dstflt_list) == 0:
		tail_entityID_dstflt_list.append(tail_entityID_list[0])

	# convert to numpy array
	head_neighbour_arr = np.array(head_entityID_dstflt_list)
	head_entityID_arr = head_neighbour_arr[:, 0].astype(np.int64)
	tail_neighbour_arr = np.array(tail_entityID_dstflt_list)
	tail_entityID_arr = tail_neighbour_arr[:, 0].astype(np.int64)
	cur_relation_arr = np.array(int(cur_relation))

	# cartesian of head entity and tail entity
	ht_pairs = cartesian([head_entityID_arr, tail_entityID_arr])
	cur_relation_arr = np.broadcast_to(cur_relation_arr, (ht_pairs.shape[0],))

	# the score based on the similarity
	Similarity_score_arr = []
	for item in ht_pairs:
		h_index = np.where(head_neighbour_arr[:, 0] == str(item[0]))[0][0]
		t_index = np.where(tail_neighbour_arr[:, 0] == str(item[1]))[0][0]
		similar_score = float(head_neighbour_arr[h_index][1]) + float(tail_neighbour_arr[t_index][1])
		Similarity_score_arr.append(similar_score)

	Similarity_score_arr = np.array(Similarity_score_arr)
	# get the embedding of entities and relations
	em_ht_pairs = KBC_model.EntityVectors.vectors[ht_pairs]
	em_relations = KBC_model.RelationVectors.vectors[cur_relation_arr]

	# the score given by discriminator
	Discriminator_score_arr = discriminator.predict([em_ht_pairs, em_relations], batch_size=batch_size * 30)

	# the final score is the element-wise product of Discriminator_score and Similarity_score
	score_arr = Similarity_score_arr * Discriminator_score_arr.reshape(Similarity_score_arr.shape[0], )

	index_max_likelihood = np.where(score_arr == np.max(score_arr))[0][0]  # the index of max value of em_ht_pairs

	score = score_arr[index_max_likelihood]  # evaluating score of discriminator
	D_score = Discriminator_score_arr[index_max_likelihood][0]
	if score >= Dtrust_threshold:  # we can trust the discriminator
		head_entityID = str(ht_pairs[index_max_likelihood, 0])
		tail_entityID = str(ht_pairs[index_max_likelihood, 1])
	else:
		# if we cannot trust the discriminator, just get the nearest neighbour
		head_entityID = head_entityID_arr[0]
		tail_entityID = tail_entityID_arr[0]
		score = 0.7  # means it is the nearest neighbour, nearest neighbour with medium priority

	Predict_Epair.append(score)
	Predict_Epair.append(D_score)
	Predict_Epair.append(head_entityID)
	Predict_Epair.append(tail_entityID)
	Predict_Epair.append(cur_relation)

	return Predict_Epair


# sample_ratio: just select sample_ratio tuples of each relation
# we figure out the similarity score and discrinimator score adn multiply them as comprehensive score
# then we select the highest one as the final result
# However, if the comprehensive score is lower than Dtrust_threshold, the nearest neighbour is used as the result
def Generate_samples_selected_Evaluate_with_Mulpy(Dtrust_threshold, similarity_threshold, sample_ratio, file_no):
	nIndex = 0
	NeighbourScope = 5  # select five most similar items from the generated point in embedding space
	EntityPairs = []
	for x_item in list(x_decoded):
		y_item = Y_sample[nIndex, :]
		Predict_Epair = EvaluateNeibor4SingleSample_Mulpy(x_item, y_item, NeighbourScope, Dtrust_threshold,
														  similarity_threshold)
		nIndex = nIndex + 1

		EntityPairs.append(Predict_Epair)

	# recompute the score of every entity
	# for item in EntityPairs:
	#     item[0] = ComputeScore_TransEFeature(item)

	# sort triplet_arr, first: relationship, second: score
	EntityPairs = np.array(EntityPairs, dtype=float)
	sort_idex = np.lexsort([-1 * EntityPairs[:, 0], EntityPairs[:, 4]])
	SortdEntityPairs = EntityPairs[sort_idex, :]

	file_name = "./computing_results/new/PredictedEntityPairs"
	if file_no == 0:
		file_name += ".txt"
	else:
		file_name = file_name + str(file_no + 1) + ".txt"
	# selected the top-k best triplets to save
	with open(file_name, "w") as PEP_file:
		EntityPairs_r = []
		cur_relation = SortdEntityPairs[0, 4]
		for item in SortdEntityPairs:
			relationID = item[4]
			if relationID != cur_relation:  # all the relaiton triplets of cur_relation is found, we can handle with it
				keep_num = int(len(EntityPairs_r) * sample_ratio)

				for i in range(keep_num):  # we just save top-keep_num triplets
					item_r = EntityPairs_r[i]
					score = str(item_r[0])
					D_score = str(item_r[1])
					head_entityMID = str(int(item_r[2]))
					tail_entityMID = str(int(item_r[3]))
					relationID = str(int(item_r[4]))
					PEP_file.write(
						score + '\t' + D_score + '\t' + head_entityMID + '\t' + tail_entityMID + '\t' + relationID + "\n")

				cur_relation = relationID
				EntityPairs_r = []

			EntityPairs_r.append(item)
		PEP_file.close()


# we figure out the similarity score and discrinimator score adn multiply them as comprehensive score
# then we select the highest one as the final result
# However, if the comprehensive score is lower than Dtrust_threshold, the nearest neighbour is used as the result
def Generate_samples_Evaluate_with_Mulpy(Dtrust_threshold=0.7, similarity_threshold=0):
	nIndex = 0
	NeighbourScope = 1  # select five most similar items from the generated point in embedding space
	EntityPairs = []
	for x_item in list(x_decoded):
		Predict_Epair = []
		head_entityID_list = KBC_model.EntityVectors.most_similar(
			x_item[0, :].reshape(1, embedding_dim), topn=NeighbourScope)

		tail_entityID_list = KBC_model.EntityVectors.most_similar(
			x_item[1, :].reshape(1, embedding_dim), topn=NeighbourScope)

		y_item = Y_sample[nIndex, :]
		cur_relation = KBC_model.RelationVectors.most_similar(y_item.reshape(1, embedding_dim), topn=1)[0][0]

		head_entityID_dstflt_list = []
		for item in head_entityID_list:
			if item[1] >= similarity_threshold:
				head_entityID_dstflt_list.append(item)

		tail_entityID_dstflt_list = []
		for item in tail_entityID_list:
			if item[1] >= similarity_threshold:
				tail_entityID_dstflt_list.append(item)

		if len(head_entityID_dstflt_list) == 0:
			head_entityID_dstflt_list.append(head_entityID_list[0])
		if len(tail_entityID_dstflt_list) == 0:
			tail_entityID_dstflt_list.append(tail_entityID_list[0])

		# convert to numpy array
		head_neighbour_arr = np.array(head_entityID_dstflt_list)
		head_entityID_arr = head_neighbour_arr[:, 0].astype(np.int64)
		tail_neighbour_arr = np.array(tail_entityID_dstflt_list)
		tail_entityID_arr = tail_neighbour_arr[:, 0].astype(np.int64)
		cur_relation_arr = np.array(int(cur_relation))

		# cartesian of head entity and tail entity
		ht_pairs = cartesian([head_entityID_arr, tail_entityID_arr])
		cur_relation_arr = np.broadcast_to(cur_relation_arr, (ht_pairs.shape[0],))

		# the score based on the similarity
		Similarity_score_arr = []
		for item in ht_pairs:
			h_index = np.where(head_neighbour_arr[:, 0] == str(item[0]))[0][0]
			t_index = np.where(tail_neighbour_arr[:, 0] == str(item[1]))[0][0]
			similar_score = float(head_neighbour_arr[h_index][1]) + float(tail_neighbour_arr[t_index][1])
			Similarity_score_arr.append(similar_score)

		Similarity_score_arr = np.array(Similarity_score_arr)
		# get the embedding of entities and relations
		em_ht_pairs = KBC_model.EntityVectors.vectors[ht_pairs]
		em_relations = KBC_model.RelationVectors.vectors[cur_relation_arr]

		# the score given by discriminator
		Discriminator_score_arr = discriminator.predict([em_ht_pairs, em_relations], batch_size=batch_size * 30)

		# the final score is the element-wise product of Discriminator_score and Similarity_score
		score_arr = Similarity_score_arr * Discriminator_score_arr.reshape(Similarity_score_arr.shape[0], )

		index_max_likelihood = np.where(score_arr == np.max(score_arr))[0][0]  # the index of max value of em_ht_pairs

		score = score_arr[index_max_likelihood]  # evaluating score of discriminator
		if score >= Dtrust_threshold:  # we can trust the discriminator
			head_entityID = str(ht_pairs[index_max_likelihood, 0])
			tail_entityID = str(ht_pairs[index_max_likelihood, 1])
		else:
			# if we cannot trust the discriminator, just get the nearest neighbour
			head_entityID = head_entityID_arr[0]
			tail_entityID = tail_entityID_arr[0]
			score = 0  # means it is the nearest neighbour

		nIndex = nIndex + 1
		Predict_Epair.append(score)
		Predict_Epair.append(head_entityID)
		Predict_Epair.append(tail_entityID)
		Predict_Epair.append(cur_relation)

		EntityPairs.append(Predict_Epair)

	# sort EntityPairs with the score
	# EntityPairs_arr = np.array(EntityPairs)
	# EntityPairs_arr.sort(axis=0)

	with open("./computing_results/new/PredictedEntityPairs.txt", "w") as PEP_file:
		for item in EntityPairs:
			score = str(item[0])
			head_entityMID = str(item[1])
			tail_entityMID = str(item[2])
			relationID = str(item[3])

			PEP_file.write(score + '\t' + head_entityMID + '\t' + tail_entityMID + '\t' + relationID + "\n")
	PEP_file.close()


# we get the similarity score and the discrinimator score
def Generate_samples_Evaluate_with_Discrinator(Dtrust_threshold=0.58, similarity_threshold=0.5):
	nIndex = 0
	NeighbourScope = 5  # select five most similar items from the generated point in embedding space
	EntityPairs = []
	for x_item in list(x_decoded):
		Predict_Epair = []
		head_entityID_list = KBC_model.EntityVectors.most_similar(
			x_item[0, :].reshape(1, embedding_dim), topn=NeighbourScope)

		tail_entityID_list = KBC_model.EntityVectors.most_similar(
			x_item[1, :].reshape(1, embedding_dim), topn=NeighbourScope)

		y_item = Y_sample[nIndex, :]
		cur_relation = KBC_model.RelationVectors.most_similar(y_item.reshape(1, embedding_dim), topn=1)[0][0]

		# we only keep the neighbours of which the distance between the generated entity is more close than distance_threshold
		head_entityID_dstflt_list = []
		for item in head_entityID_list:
			if item[1] >= similarity_threshold:
				head_entityID_dstflt_list.append(item)

		tail_entityID_dstflt_list = []
		for item in tail_entityID_list:
			if item[1] >= similarity_threshold:
				tail_entityID_dstflt_list.append(item)

		# if we don't have any entity close than than distance_threshold,just keep the closest one.
		if len(head_entityID_dstflt_list) == 0:
			head_entityID_dstflt_list.append(head_entityID_list[0])
		if len(tail_entityID_dstflt_list) == 0:
			tail_entityID_dstflt_list.append(tail_entityID_list[0])

		# convert to numpy array
		head_entityID_arr = np.array(head_entityID_dstflt_list)
		head_entityID_arr = head_entityID_arr[:, 0].astype(np.int64)
		tail_entityID_arr = np.array(tail_entityID_dstflt_list)
		tail_entityID_arr = tail_entityID_arr[:, 0].astype(np.int64)
		cur_relation_arr = np.array(int(cur_relation))

		# cartesian of head entity and tail entity
		ht_pairs = cartesian([head_entityID_arr, tail_entityID_arr])
		cur_relation_arr = np.broadcast_to(cur_relation_arr, (ht_pairs.shape[0],))

		# get the embedding of entities and relations
		em_ht_pairs = KBC_model.EntityVectors.vectors[ht_pairs]
		em_relations = KBC_model.RelationVectors.vectors[cur_relation_arr]

		valid_flag = discriminator.predict([em_ht_pairs, em_relations], batch_size=batch_size * 30)
		index_max_likelihood = np.where(valid_flag == np.max(valid_flag))[0][0]  # the index of max value of em_ht_pairs

		score = valid_flag[index_max_likelihood][0]  # evaluating score of discriminator
		if score >= Dtrust_threshold:  # we can trust the discriminator
			head_entityID = str(ht_pairs[index_max_likelihood, 0])
			tail_entityID = str(ht_pairs[index_max_likelihood, 1])
		else:
			# if we cannot trust the discriminator, just get the nearest neighbour
			head_entityID = head_entityID_arr[0]
			tail_entityID = tail_entityID_arr[0]
			score = 0  # means it is the nearest neighbour

		nIndex = nIndex + 1
		Predict_Epair.append(score)
		Predict_Epair.append(head_entityID)
		Predict_Epair.append(tail_entityID)
		Predict_Epair.append(cur_relation)

		EntityPairs.append(Predict_Epair)

	# sort EntityPairs with the score
	# EntityPairs_arr = np.array(EntityPairs)
	# EntityPairs_arr.sort(axis=0)

	with open("./computing_results/new/PredictedEntityPairs.txt", "w") as PEP_file:
		for item in EntityPairs:
			score = str(item[0])
			head_entityMID = str(item[1])
			tail_entityMID = str(item[2])
			relationID = str(item[3])

			PEP_file.write(score + '\t' + head_entityMID + '\t' + tail_entityMID + '\t' + relationID + "\n")
	PEP_file.close()


# return the most frequent top n elements from array arr
def counter(arr, top_n):
	return Counter(arr).most_common(top_n)


# using h+r-t=0 to evaluate the candidate pairs
def ComputeScore_TransEFeature(entity_pair):
	h = int(entity_pair[2])
	t = int(entity_pair[3])
	r = int(entity_pair[4])

	em_head = KBC_model.EntityVectors.vectors[h]
	em_tail = KBC_model.EntityVectors.vectors[t]
	em_relation = KBC_model.RelationVectors.vectors[r]

	score = -np.sum(np.square(em_head + em_relation - em_tail))
	return score


if __name__ == '__main__':
	# data file path
	if dataset_flag == "FB15K":		#the data set is FB15K
		training_data_file_name = os.getcwd() + '/data/FB15K_TransE/train2id_Selected200_1.txt'
		validation_data_file_name = os.getcwd() + '/data/FB15K_TransE/valid2id_Selected200.txt'
		test_data_file_name = os.getcwd() + '/data/FB15K_TransE/test2id_Selected200.txt'
		# embdding
		entity_embedding_file_name = './data/FB15K_TransE/entity_embeddings.npy'
		relation_embedding_file_name = './data/FB15K_TransE/relation_embeddings.npy'

	else:			#the data set is WN18
		training_data_file_name = os.getcwd() + '/data/WN18_TransE/train2id_Selected200_1.txt'
		validation_data_file_name = os.getcwd() + '/data/WN18_TransE/valid2id_Selected200.txt'
		test_data_file_name = os.getcwd() + '/data/WN18_TransE/test2id_Selected200.txt'
		# embdding
		entity_embedding_file_name = './data/WN18_TransE/entity_embeddings.npy'
		relation_embedding_file_name = './data/WN18_TransE/relation_embeddings.npy'

	# Load data
	data_file_name_list = [training_data_file_name, validation_data_file_name, test_data_file_name]
	X_data_list, Y_data_list = Load_all_data(data_file_name_list)

	# define data list
	X_train = np.array(X_data_list[0])
	Y_train = np.array(Y_data_list[0])
	X_validation = np.array(X_data_list[1])
	Y_validation = np.array(Y_data_list[1])
	X_test = np.array(X_data_list[2])
	Y_test = np.array(Y_data_list[2])

	#construct the KGGen model
	KBC_model = VAE_GAN(entity_embedding_file_name, relation_embedding_file_name)

	# normalize input embeddings
	# np_X_train = embedding_rescale(KBC_model.embeddinglayer.predict(X_train, batch_size=batch_size), embedding_scale)
	# np_X_Validation = embedding_rescale(KBC_model.embeddinglayer.predict(X_validation, batch_size=batch_size),
	#                                     embedding_scale)
	# np_X_test = embedding_rescale(KBC_model.embeddinglayer.predict(X_test, batch_size=batch_size), embedding_scale)

	# get the embeddings of entity
	np_X_train = embedding_rescale(KBC_model.EntityVectors.vectors[X_train], embedding_scale)
	np_X_Validation = embedding_rescale(KBC_model.EntityVectors.vectors[X_validation], embedding_scale)
	np_X_test = embedding_rescale(KBC_model.EntityVectors.vectors[X_test], embedding_scale)

	# random generate np_X_train and np_X_Validation for debug. the embedding process is too slow
	# np_X_train = np.random.random([X_train.shape[0], 2, 100])
	# np_X_Validation = np.random.random([X_validation.shape[0], 2, 100])
	# np_X_test = np.random.random([X_test.shape[0], 2, 100])

	# get the embeddings of relationship
	np_Y_train = KBC_model.RelationVectors.vectors[Y_train]
	np_Y_Validation = KBC_model.RelationVectors.vectors[Y_validation]
	np_Y_test = KBC_model.RelationVectors.vectors[Y_test]

	# get unique Y_set
	Y = np.concatenate(Y_data_list, axis=0)
	Y_lable_unique = np.array(list(set(Y)))  # The unique item in Y label
	Y_lable_unique.sort()

	# Create the dictionary for computing results
	mkpath = "./computing_results/new/fig"
	mkdir(mkpath)

	mkpath = "./model"
	mkdir(mkpath)

	# ------------------Build Discriminator network to train D---------------------
	optimizer = Adam(0.00003, 0.5, decay=1e-4)
	discriminator = KBC_model.BuildDisctriminator()
	discriminator.compile(loss='binary_crossentropy',
						  optimizer=optimizer,
						  metrics=['accuracy'])

	# ------------------Build combined network to train VAE(E and G)---------------------
	encoder = KBC_model.BuildEncoder()
	generator = KBC_model.BuildDecoder()

	embed_input = Input((2, embedding_dim,))
	# sequence_input = Input(shape=(None,))
	relation_input = Input((relationship_dim,))
	z_mean = encoder([embed_input, relation_input])[0]
	z_log_var = encoder([embed_input, relation_input])[1]
	z_cond = encoder([embed_input, relation_input])[2]
	entity_pairs_g = generator(z_cond)

	# For the combined model we will only train the generator
	discriminator.trainable = False

	# The discriminator takes generated htr as input and determines validity
	validity_p = discriminator([entity_pairs_g, relation_input])

	# Trains the generator to fool the discriminator
	combined = Model([embed_input, relation_input], validity_p)

	# The combined model  (stacked generator and discriminator)
	validity_r = Input((1,))
	combined_loss_layer = CustomCombinedNNLossLayer()
	combined_loss = combined_loss_layer([embed_input, entity_pairs_g, validity_r, validity_p])
	combined = Model([embed_input, relation_input, validity_r], combined_loss)
	combined.compile(optimizer=Adam(0.0001, 0.5, decay=1e-4), loss=None)  # decay the learning rate
	combined.summary()
	# plot_model(combined, to_file='VAE_GAN.png', show_shapes=True)

	# filepath = "./model/NN_para_{epoch:02d}.hdf5"
	# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	#
	# #call back function will be called at each e
	# callbacks_list = [AneelingCallback(anneal_schedule, hp_lambda), checkpoint, tbCallBack]

	# # fit the model
	# history = vae.fit([np_X_train, Y_train_onehot],
	#                   shuffle=True,
	#                   epochs=vae_epochs,
	#                   batch_size=batch_size,
	#                   validation_data=([np_X_Validation, Y_validation_onehot], None),
	#                   callbacks=callbacks_list)  # call fucntions in callbacks_list in every epoch
	#
	# vae.save_weights("./model/VAE.hdf5")
	# # encoder.save_weights("./model/encoder.hdf5")
	# encoder.save("./model/encoder.h5")

	# ------------------------------------------------------------------------------------------------------
	if load_weights == True:
		generator.load_weights("./model/Generator.hdf5")
		discriminator.load_weights("./model/discriminator.hdf5")
		encoder.load_weights("./model/encoder.hdf5")

	BtSample = BatchSampling(np_X_train.shape[0])  # the class for sampling block by block

	if training_flag == True:
		# ---------------------------------------train VAE_GAN network----------------------------------------------
		CurRcLossValidation = 1000
		bEarlyTerminatedFlag = False
		nNoProgress = 0  # the iteration with no progress
		for epoch in range(epochs):

			if bEarlyTerminatedFlag == True:  # so we need early terminate the training process
				break

			if (epoch >= Discriminator_Start_epoch):
				# ---------------------
				#  Train Discriminator
				# ---------------------

				# Select a random batch of Training data
				idx = np.random.randint(0, np_X_train.shape[0], batch_size // 2)
				real_ht = np_X_train[idx]
				real_relation = np_Y_train[idx]

				# Generate the fake samples
				# sampling form z-space
				z_random_vector, gen_relation1 = SampleFormSpaceZ(batch_size // 4, latent_dim, np_Y_train)
				# Generate a batch of new entity pairs
				gen_ht1 = generator.predict(z_random_vector)

				# trans from the whole network
				idx = np.random.randint(0, np_X_train.shape[0], batch_size // 4)
				real_ht_toNN = np_X_train[idx]
				real_relation_toNN = np_Y_train[idx]
				# z_vector_mean = (encoder.predict([real_ht_toNN, real_relation_toNN]))[0]
				# z_vector_fromX = np.hstack((z_vector_mean,real_relation_toNN))  # concatenate the r_ht and r_relation_onehot(the input of decoder)
				z_vector_fromX = (encoder.predict([real_ht_toNN, real_relation_toNN]))[2]
				gen_ht2 = generator.predict(z_vector_fromX)

				gen_ht = np.concatenate((gen_ht1, gen_ht2), axis=0)
				gen_relation = np.concatenate((gen_relation1, real_relation_toNN), axis=0)

				# generate labels
				valid = np.ones((batch_size // 2, 1))
				fake = np.zeros((batch_size // 2, 1))

				if epoch % 5 == 0:  # every 5 Iterations, we update the discriminator
					# Train the discriminator
					d_loss_real = discriminator.train_on_batch([real_ht, real_relation], valid)
					d_loss_fake = discriminator.train_on_batch([gen_ht, gen_relation], fake)
					d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
				else:
					# Test the discriminator
					d_loss_real = discriminator.test_on_batch([real_ht, real_relation], valid)
					d_loss_fake = discriminator.test_on_batch([gen_ht, gen_relation], fake)
					d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

					# ---------------------
			#  Train Generator(in fact, VAE- the encoder and generator)
			# ---------------------

			# sampling form X_training data
			# Select a random batch of Training data
			# idx = np.random.randint(0, np_X_train.shape[0], batch_size)
			# sample_ht = np_X_train[idx]
			# sample_relation = np_Y_train[idx]

			idx, ShuffleFlag = BtSample.BlockSampling_index(batch_size)  # sampling block bu block
			sample_ht = np_X_train[idx]
			sample_relation = np_Y_train[idx]

			if ShuffleFlag == True:  # shaffle the training data set
				np_X_train, np_Y_train = shuffle_in_unison(np_X_train, np_Y_train)

			c_valid = np.ones((batch_size, 1))
			# Train the generator (to have the discriminator label samples as valid)
			g_loss = combined.train_on_batch([sample_ht, sample_relation, c_valid], None)

			# Plot the progress
			if (epoch < Discriminator_Start_epoch):
				print("%d [D loss: not started] [G loss: %f]" % (epoch, g_loss))
			else:
				print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

			if epoch % 100000 == 0:
				strFileName = "./computing_results/new/fig/LatentSpace_trainingdata" + str(epoch) + ".pdf"
				VisulaizelatentSpace(np_X_train, np_Y_train, Y_train, batch_size, strFileName)

				# strFileName = "./computing_results/new/fig/LatentSpace_trainingdata" + str(epoch)
				# VisulaizelatentSpace4LargeData(np_X_train, np_Y_train, Y_train, batch_size, strFileName)

				# strFileName = "./computing_results/new/fig/EmbeddingSpace_testdata" + str(epoch) + ".pdf"
				# VisulaizeEmbeddingSpace(np_X_test, np_Y_test, Y_test, strFileName)
				#
				# strFileName = "./computing_results/new/fig/EmbeddingSpace_traindata" + str(epoch) + ".pdf"
				# VisulaizeEmbeddingSpace(np_X_train, np_Y_train, Y_train, strFileName)

				c_valid = np.ones((np_X_train.shape[0], 1))
				generation_loss = combined.test_on_batch([np_X_train, np_Y_train, c_valid], None)
				print("\n the generation error of training data is " + str(generation_loss) + "\n")
				c_valid = np.ones((np_X_Validation.shape[0], 1))
				generation_loss = combined.test_on_batch([np_X_Validation, np_Y_Validation, c_valid], None)
				print("\n the generation error of validation data is " + str(generation_loss) + "\n")
				c_valid = np.ones((np_X_test.shape[0], 1))
				generation_loss = combined.test_on_batch([np_X_test, np_Y_test, c_valid], None)
				print("\n the generation error of test data is " + str(generation_loss) + "\n")

				# reconstuction_loss = ComputingReconstructionLoss(np_X_train, np_Y_train)
				# print "\n the reconstruction error of training data is " + str(reconstuction_loss) + "\n"
				reconstuction_loss = ComputingReconstructionLoss(np_X_Validation, np_Y_Validation)
				print("\n the reconstruction error of validation data is " + str(reconstuction_loss) + "\n")
				# reconstuction_loss = ComputingReconstructionLoss(np_X_test, np_Y_test)
				# print "\n the reconstruction error of test data is " + str(reconstuction_loss) + "\n"

			# dy_Hvalue = min(epoch * 0.000025, 0.03)
			dy_Hvalue = min(epoch * 0.000002, 0.01)
			K.set_value(hp_lambda, dy_Hvalue)

			# start the discriminator, and change the loss function
			if epoch == Discriminator_Start_epoch:
				K.set_value(d_lambda, 0.2)

			if epoch >= 600000:
				nNoProgress = nNoProgress + 1
				if (epoch >= Discriminator_Start_epoch and epoch % 1 == 0):

					# cross validation
					# if (NoProgress_Iteration4EarlyTerminate - nNoProgress > 1500):
					#     RcLossValidation = ComputingReconstructionLoss_RandomBatch(np_X_Validation, np_Y_Validation, 1000)
					# else:
					#     RcLossValidation = ComputingReconstructionLoss(np_X_Validation, np_Y_Validation)

					RcLossValidation = ComputingReconstructionLoss(np_X_Validation, np_Y_Validation)
					if RcLossValidation < CurRcLossValidation:
						print("Reconstruction Loss of Validate data improves from %f to %f" % (
							CurRcLossValidation, RcLossValidation))
						CurRcLossValidation = RcLossValidation
						nNoProgress = 0

						# generator.save("./model/Generator.h5")
						generator.save_weights("./model/Generator.hdf5")
						discriminator.save_weights("./model/discriminator.hdf5")
						encoder.save_weights("./model/encoder.hdf5")

					# cross validation for generation loss
					# RcLossValidation = ComputingGenerationLoss_RandomBatch(np_X_Validation, np_Y_Validation, 7000)

					# if RcLossValidation < CurRcLossValidation:
					#     print("Loss of validation data improves from %f to %f" % (
					#     CurRcLossValidation, RcLossValidation))
					#     CurRcLossValidation = RcLossValidation
					#     nNoProgress = 0
					#
					#     generator.save("./model/Generator.h5")

			if (epoch >= Discriminator_Start_epoch and nNoProgress >= NoProgress_Iteration4EarlyTerminate):
				bEarlyTerminatedFlag = True
		# combined.save_weights("./model/GAN.hdf5")
		# decoder.save_weights("./model/Generatorpara.hdf5")
		# generator.save("./model/Generator.h5")

		# ------------------------------------------------------------------------------------------------------
	else:  # load pretrained NN model parameters
		# encoder = load_model("./model/encoder.h5")
		# generator = load_model("./model/Generator_TransE_sports_(120000).h5")

		generator.load_weights("./model/Generator.hdf5")
		discriminator.load_weights("./model/discriminator.hdf5")
		encoder.load_weights("./model/encoder.hdf5")
		print("Loading model data, done\n")
	# ---------------------------------------Evaluation the results-----------------------------------------

	if drawing_picture == True:
		reconstuction_loss = ComputingReconstructionLoss(np_X_train, np_Y_train)
		print("\n the reconstruction error of training data is " + str(reconstuction_loss) + "\n")
		reconstuction_loss = ComputingReconstructionLoss(np_X_Validation, np_Y_Validation)
		print("\n the reconstruction error of validation data is " + str(reconstuction_loss) + "\n")
		reconstuction_loss = ComputingReconstructionLoss(np_X_test, np_Y_test)
		print("\n the reconstruction error of test data is " + str(reconstuction_loss) + "\n")

		# display a 2D plot of the digit classes in the latent space
		VisulaizelatentSpace_withsamplingdata(np_X_test, np_Y_test, Y_test, batch_size,
											  "./computing_results/new/fig/LatentSpace_testdata.pdf", Z_sample)

		VisulaizelatentSpace_withsamplingdata(np_X_train, np_Y_train, Y_train, batch_size,
											  "./computing_results/new/fig/LatentSpace_trainingdata.pdf", Z_sample)

		VisulaizelatentSpace_withsamplingdata(np_X_Validation, np_Y_Validation, Y_validation, batch_size,
											  "./computing_results/new/fig/LatentSpace_validationdata.pdf", Z_sample)

		# VisulaizelatentSpace4sampling(Z_sample, "./computing_results/new/fig/LatentSpace_sampledata.pdf")

		VisulaizeEmbeddingSpace(np_X_test, np_Y_test, Y_test,
								"./computing_results/new/fig/EmbeddingSpace_testdata.pdf")

		VisulaizeEmbeddingSpace(np_X_train, np_Y_train, Y_train,
								"./computing_results/new/fig/EmbeddingSpace_traindata.pdf")

		VisulaizeEmbeddingSpace(np_X_Validation, np_Y_Validation, Y_validation,
								"./computing_results/new/fig/EmbeddingSpace_validationdata.pdf")

		print("\nLatent Space and embedding space visulization is done\n")

	theTime = datetime.datetime.now()
	Msg = "Start Time: " + str(theTime)
	print(Msg)

	for i in range(result_files_number):
		# sampling based on the distribution fitting
		X_train_encoded_all = encoder.predict([np_X_train, np_Y_train], batch_size=batch_size)
		X_train_encoded = X_train_encoded_all[0]

		# X_train_encoded = np.random.random([X_train_encoded.shape[0], 200])
		# Y_train = np.random.random_integers(0,1039,X_train_encoded.shape[0])

		# Y_train = np.nan_to_num(Y_train)

		# //////////////////test the distribution of x_test data/////////////////////////////////////
		# X_test_encoded_all = encoder.predict([np_X_test, Y_test_onehot], batch_size=batch_size)
		# Z_sample_test = X_test_encoded_all[0]
		# Y_sample_test = Y_test_onehot
		# x_decoded_test = generator.predict(np.concatenate([Z_sample_test, Y_sample_test], axis=1))
		# ///////////////////////////////////////////////////////////////////////////////////////////

		# Z_sample = X_train_encoded_all[0]
		# Y_sample = Y_train_onehot

		# ramdonly generate same number samples
		# x_decoded, Z_sample, Y_sample = DenseSamplingFromZ_Snum(kde_sample_num)      #dense sampling the same number samples from each relationship

		# ramdonly generate samples with percentage over total triplets with the a relationship
		RelationFreq_list = counter(Y_train, len(Y_lable_unique))
		x_decoded, Z_sample, Y_sample = DenseSamplingFromZ_SRatio(sample_ratio_gross, RelationFreq_list)

		Msg = "randomly generate sample data, done, step: " + str(i + 1)
		print(Msg)

		# generate and evaluate samples with the comprehensive score, filter samples with similarity score
		Dtrust_threshold = 0.7  # The threshold of trusting discrinimator
		similarity_threshold = 0  # The threshold of similarity of top-k neighbour

		# evaluate the samples and select a subset from generated samples with the sample_ratio_fine
		Generate_samples_selected_Evaluate_with_Mulpy(Dtrust_threshold, similarity_threshold, sample_ratio_fine, i)

		# just evaluate the samples
		# Generate_samples_Evaluate_with_Mulpy(Dtrust_threshold, similarity_threshold)

		# generate and evaluate samples with the discriminator score, filter samples with similarity score
		# Dtrust_threshold = 0.58  # The threshold of trusting discrinimator
		# similarity_threshold = 0.5  # The threshold of similarity of top-k neighbour
		# Generate_samples_Evaluate_with_Discrinator(Dtrust_threshold, similarity_threshold)

		Msg = "file " + str(i + 1) + " is generatd"
		print(Msg)

	print("evaluate and select sampled data, done")
	theTime = datetime.datetime.now()
	Msg = "End Time: " + str(theTime)
	print(Msg)
