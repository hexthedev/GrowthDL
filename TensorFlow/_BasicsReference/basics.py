from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ref_tensors as tensors
import ref_sessions as sessions

# tensors.execute()
# sessions.execute()












































if False:
    
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
