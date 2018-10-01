from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

    
def write_event_file():
    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())
    print("Event Written")


def execute():
    """
    Tensor programs of graphs made of operations and tensors

    Tensors: N dimensional arrays
    Operations: Some kinf of transformation that normally outputs a tensor

    Below are examples of a constant operation that outputs a single dimensional vector.
    
    If you try to print these lines, you won't get a value because they have not yet been run in a session
    """  
    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0) # also tf.float32 implicitly
    
    
    
    """
    We can create a graph by doing operations.
    """
    total = a + b


    """
    You can easily visualize this graph by using tensor board. 

    1) Output event file using write_event_file() [ABOVE]
    2) Use command tensorboard --logdir . in your terminal

    It'll start a webapp at localhost:6006
    """
