import numpy as np
import tensorflow as tf
import time as t

def tensors():
    
    print('''
Tensorflow programs are graphs
Graphs are made up of Operations and Tensors
    ''')
    input()

    print('''
Consider the following code: 

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
    ''')
    input()

    print('''
What you're seeing here is the creation of a graph

a = tf.constant(3.0, dtype=tf.float32) 
--> is an operation that outputs a tensor

total = a + b
--> is a node later in the graph that takes in the tensors output by a and b
    ''')
    input()

    print('''
Lets try printing these now:
    ''')

    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0) # also tf.float32 implicitly
    total = a + b

    print(a)
    print(b)
    print(total)

    input()

    print('''
As you can see they have no value. This is because the graph has not been run. 
It has just been put together
    ''')

    input()

    print('''
END OF SESSION
    ''')
