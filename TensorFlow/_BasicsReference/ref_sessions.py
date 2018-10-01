from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def execute():
    """
    A Tensorflow session is used to run a graph
    """
    sess = tf.Session()

    """
    When you run something in a session, tensorflow will evaulate the object based on the graph
    """
    a = tf.constant(2.0, dtype=tf.float32)
    b = tf.constant(3.0)

    total = a+b

    sess.run(total)


    """
    You can do this cool thing where you pass an object to the session

    this will return the evaulations in the object
    """
    sess.run( { 'ab':(a,b), 'total':total } )


    """
    In a single run, variables will remain conistent. Between runs this is not the case.
    """
    vec = tf.random_uniform(shape=(3, ))
    out1 = vec + 1
    out2 = vec + 2

    sess.run(out1)
    sess.run(out2)
    sess.run((out1, out2))
