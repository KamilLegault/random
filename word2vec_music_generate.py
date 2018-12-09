# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code based on TensorFlow word2vec example, modified by Dorien Herremans as described in
# Herremans D., Chuan C.H. Modeling Musical Context with Word2vec. Proceedings of the International Workshop on Deep Learning and Music. Anchorage, Alaska. May 18-19, 2017.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import csv
from collections import OrderedDict


num_steps = 1  # Dorien: here: it's the number of new slices you want to generate



# Step 4: Build and a skip-gram model; it is then loaded from the saved file, not trained again
vocabulary_size = 14315

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.




# Read the data into a list of strings.
def read_data():
  """Extract the first file enclosed in a zip file as a list of words"""
  file_name = 'all_encoding.txt'

  with open(file_name, 'r') as f:
      raw_data = f.read()
      print("Data length:", len(raw_data))

  data = raw_data.split(",")

  return data


words = read_data()




def build_dataset(words, vocabulary_size):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))


  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
valid_examples_raw = list(set(words))  # unique words
valid_examples = [dictionary[i] for i in valid_examples_raw]


del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
# print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
# print('test_not coded: ', valid_examples_raw[0])
# print('test_coded: ', valid_examples[0])
# print('test_encoded: ', dictionary[valid_examples_raw[0]])
# print('test_reversed: ', reverse_dictionary[valid_examples[0]])


graph = tf.Graph()


with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      normalized_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.




#LOAD THE SAVED MODEL

with tf.Session(graph=graph) as session:
  saver = tf.train.Saver()
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  saver.restore(session, "saves/word2vec_music")

  average_loss = 0
  for step in xrange(num_steps):


    sim = similarity.eval()

    sim_values =[[0,0,0]]

    # COMPUTATIONALLY EXPENSIVE: write a text file with the cosine similarities between elements
    # write to a file, all similarity values:
    #
    # # lengthx= len(sim)
    #
    # for i in range(len(sim)):
    #   for j in range(len(sim[i])):
    #     # if reverse_dictionary[i] == '163':
    #     #     if reverse_dictionary[j] == '5869':
    #     #       print('here');
    #     # i = dictionary['5869']
    #     # j = dictionary['163']
    #     if int(i) <= int(j):
    #       sim_values.append([reverse_dictionary[i], reverse_dictionary[j], sim[i][j]])
    #     # else: print ('not: ', i, j)


    # with open("similarity_word2vec.csv", 'w') as f:
    #   writer = csv.writer(f)
    #   writer.writerows(sim_values[1:])





    # not needed, only for validation:
    # for i in xrange(valid_size):
    #     valid_word = reverse_dictionary[valid_examples[i]]
    #     # valid_word =  valid_examples[i]  #reverse_dictionary[valid_examples[i]]
    #     top_k = vocabulary_size  # number of nearest neighbor
    #     nearest = (sim[i, :])   #.argsort()[1:top_k + 1]
    #     nearest_test = (-sim[i, :]).argsort()[1:top_k + 1]
    #     #log_str = "Nearest to %s:" % valid_word
    #     for k in xrange(top_k):
    #
    #       close_word = reverse_dictionary[nearest[k]]
    #       log_str = "%s %s," % (log_str, close_word)
    #       print ("word ", close_word, " with distance ", nearest[k], 'position ', k)
    #     print(log_str)

    #
    # for i in xrange(valid_size):
    #   # valid_word = reverse_dictionary[valid_examples[i]]
    #   valid_word = reverse_dictionary[valid_examples[i]]
    #   top_k = 8  # number of nearest neighbors
    #   nearest = (-sim[i, :]).argsort()[1:top_k + 1]
    #   log_str = "Nearest to %s:" % valid_word
    #   for k in xrange(top_k):
    #     close_word = reverse_dictionary[nearest[k]]
    #     log_str = "%s %s," % (log_str, close_word)
    #   print(log_str)


  final_embeddings = normalized_embeddings.eval()






# FOR finding the similarity between PAIRS (text file with entries: word, word per line:
#
# def write_sim_pairs(filename):
#
#     simpairs = read_examples(filename)
#     simresults = [sim[dictionary[i[0]], dictionary[i[1]]] for i in simpairs]
#     resultfilename = 'sim_' + filename
#
#     f = open(resultfilename, 'w')
#     for i in range(len(simpairs)):
#         f.write(simpairs[i][0])
#         f.write(',')
#         f.write(simpairs[i][1])
#         f.write(',')
#         f.write(str(simresults[i]))
#         f.write('\n')  # python will convert \n to os.linesep
#     f.close()


# Calculate similarity for each of the files
# for thisfilename in os.listdir('workshop_abstract/triad_pairs'):
#     write_sim_pairs(thisfilename)






# FOR VISUALISATION of the reduced vector space: each file in the vis directory contains lines: word, label

# Step 6: Visualize the embeddings.

def plot_with_labels(low_dim_embs, labels, colors, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(8, 8))  # in inches
  # t = np.arange(100)
  print("startplot")

  xcoord = low_dim_embs[:, 0]
  ycoord = low_dim_embs[:, 1]
  # Vega20   winter spectral gist_ncar
  plt.scatter(xcoord, ycoord, c=colors, cmap="winter", label=labels)

  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    # plt.scatter(x, y, color=mycolors)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)



def read_examples(filename):

  newdata = []

  with open(filename) as csvfile:
      readCSV = csv.reader(csvfile, delimiter=',')
      for row in readCSV:
          newdata.append(row)


  return newdata


def read_visfiles(directory):

  newdata = []

  for thisfilename in os.listdir(directory):

      filenamefull = 'vis/'+ thisfilename
      with open(filenamefull) as csvfile:
          chordname = thisfilename.split('.')[0]
          readCSV = csv.reader(csvfile, delimiter=',')
          for row in readCSV:
              newdata.append([row[0], chordname])

  return newdata


pairs = read_visfiles('vis')

# test = final_embeddings[:500, :]
encoded_pairs = [dictionary[i[0]] for i in pairs]
encoded_pairs = [dictionary[i[0]] for i in pairs]
labels = [i[1] for i in pairs]
vis_final_embeddings = [final_embeddings[i] for i in encoded_pairs]




def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color



try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  import matplotlib.cm as cmx
  import matplotlib.colors as colors


  colors = [{ni: indi for indi, ni in enumerate(set(labels))}[ni] for ni in labels]



  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=10000)
  low_dim_embs = tsne.fit_transform(vis_final_embeddings)

  plot_with_labels(low_dim_embs, labels, colors)

except ImportError:
  print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

