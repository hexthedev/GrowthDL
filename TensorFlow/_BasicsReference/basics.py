from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tutorials as tuts
import time as t

def help():
    return '''
    help: shows options
    quit: ends session
    tensors: lesson on tensors and operations. Most basic things in tensorflow
    '''

# Options: TO (Tensors and Operations)

print('''
Welcome to James Personal TensorFlow Reference.
Type the name of a tutorial to get a quick command prompt tutorial on the subject

To see commands, type HELP
''')

options = ["help", "quit", "tensors"]

commands = {
    "help": lambda : print(help()),
    "tensors": lambda : tuts.tensors()
}

while True:
    inp = input(">> ")

    if not inp in options:
        print(">> INVALID OPTION")
        commands["help"]()
        continue
    
    if inp == "quit":
        break

    commands[inp]()
    









































if False:
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
    print("\n\n-------- Session Run\n\n{0}\n\n---------\n\n".format(sess.run(total)))

    # You can also pass an evaulation object which will run each thing in the object
    # It returns a result in a structure of the same layout
    print("\n\n-------- Session Run Object\n\n{0}\n\n---------\n\n".format(sess.run({'ab': (a, b), 'total': total})))

    # Tensors in tf.Session.run calls only have a single value.
    vec = tf.random_uniform(shape=(3, ))
    out1 = vec + 1
    out2 = vec + 2

    print("\n\n-------- Session Value between and in runs\n\n{0}\n{1}\n{2}\n\n---------\n\n".format(
        sess.run(out1),
        sess.run(out2),
        sess.run((out1, out2))))


    # TF PLACEHOLDERS ------------
    # External inputs to graphs are placeholders
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = x + y

    print("\n\n-------- Placeholders\n\n{0}\n{1}\n\n---------\n\n".format(
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

    print("\n\n-------- \n\n")


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

    print("\n\n-------- \n\n")


    # Layers are used to add trainable parameters to a graph
    print("\n\n-------- Layers\n\n")

    x = tf.placeholder(tf.float32, shape=[None, 3])
    linear_model = tf.layers.Dense(units=1)
    y = linear_model(x)

    #layers contain vars that require initilaization. You can do it individually but there is also this function. 
    init = tf.global_variables_initializer() #Initializes variables in the graph
    sess.run(init)

    print(sess.run(y, {x: [[1,2,3], [4,5,6]]}))

    # Tensorflow provides shorcuts. 
    '''
    linear_model = tf.layers.Dense(units=1)
    y = linear_model(x)

    Can also be this (dense is the shorcut for Dense)
    y = ts.layers.dense(x, units=1)

    this approach hides access to Layer object though
    '''

    print("\n\n-------- \n\n")

    #Feature column organization. The department words need to be translated into numbers
    print("\n\n-------- Features\n\n")

    features = {
        'sales' : [[5], [10], [8], [9]],
        'department': ['sports', 'sports', 'gardening', 'gardening']}

    department_column = tf.feature_column.categorical_column_with_vocabulary_list(
            'department', ['sports', 'gardening'])
    department_column = tf.feature_column.indicator_column(department_column)

    columns = [
        tf.feature_column.numeric_column('sales'),
        department_column
    ]

    inputs = tf.feature_column.input_layer(features, columns)

    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()
    sess = tf.Session()
    sess.run((var_init, table_init))

    print(sess.run(inputs))

    print("\n\n-------- \n\n")


    # Training
    print("\n\n-------- Training\n\n")
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

    linear_model = tf.layers.Dense(units=1)
    y_pred = linear_model(x)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(y_pred))

    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    print(sess.run(loss))

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    for i in range(100):
        _, loss_value = sess.run((train, loss))
        print(loss_value)


    print("\n\n-------- \n\n")


    write_event_file()
