from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Tensorflow programs of Tensorflow graphs
# a graph is series of Operations and Tensors

# TENSORS AND OPERATIONS ------------
# You can make some really basic operations and put them into a graph
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b

print("\n\n------ GRAPH BASICS\n\n{0}\n{1}\n{2}\n\n-------- FIN\n\n".format(a, b, total))

# You'll see the following, because these have no value until they're run. You've just made a graph, that's it
'''
Tensor("Const:0", shape=(), dtype=float32)
Tensor("Const_1:0", shape=(), dtype=float32)
Tensor("add:0", shape=(), dtype=float32)
'''


# TENSOR BOARD EVENT FILES AND USAGE ------------
# You can run the following command to make an event file
def write_event_file():
    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())
    print("Event Written")

'''
You'll get a file with name events.out.tfevents.{timestamp}.{hostname}

Use the command: 
tensorboard --logdir .

It'll start a webapp at localhost:6006
'''


# TF SESSIONS ------------
# Tensor evaulation requires a session
sess = tf.Session()

# Sessions can be told to run a node, in which case the node graph will be evaulated as needed
print("\n\n-------- Session Run\n\n{0}\n\n---------FIN\n\n".format(sess.run(total)))

# You can also pass an evaulation object which will run each thing in the object
# It returns a result in a structure of the same layout
print("\n\n-------- Session Run Object\n\n{0}\n\n---------FIN\n\n".format(sess.run({'ab': (a, b), 'total': total})))

# Tensors in tf.Session.run calls only have a single value.
vec = tf.random_uniform(shape=(3, ))
out1 = vec + 1
out2 = vec + 2

print("\n\n-------- Session Value between and in runs\n\n{0}\n{1}\n{2}\n\n---------FIN\n\n".format(
    sess.run(out1),
    sess.run(out2),
    sess.run((out1, out2))))


# TF PLACEHOLDERS ------------
# External inputs to graphs are placeholders
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

print("\n\n-------- Placeholders\n\n{0}\n{1}\n\n---------FIN\n\n".format(
    sess.run(z, feed_dict={x:3, y:4.5}),
    sess.run(z, feed_dict={x: [1, 3], y: [2, 4]})))

'''
feed_dict can override any tensor in the graph

Only difference is that placeholders error if not fed
'''


# TF DATASETS ------------
# Datasets are the preferred method for streaming data into a model
my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]

slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

print("\n\n-------- Dataset\n\n")

while True:
  try:
    print(sess.run(next_item))
  except tf.errors.OutOfRangeError:
    break

print("\n\n-------- FIN\n\n")


# Sometimes you might to initialize the iterator before use
r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

print("\n\n-------- Initilaization\n\n")

sess.run(iterator.initializer)

while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    break

print("\n\n-------- FIN\n\n")





write_event_file()
