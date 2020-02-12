from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow.compat.v1 as tf
with tf.compat.v1.Session() as sess:

    # hello, world
    hello = tf.constant("Hello, TensorFlow!")
    print(sess.run(hello))

    # Computational graph
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)
    node3 = tf.add(node1, node2)

    print("node1:", node1, "node2:", node2)
    print("node3:", node3)

    sess.run([node1, node2])
    sess.run(node3)

    # Placeholder
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)

    adder_node = a + b
    print(sess.run(adder_node, feed_dict={a: 3, b:4.5}))
    print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

    # Ranks, Shapes and Types
    # Ranks: scalar, vector, matrix, 3-tensor
    # Shapes: 3x3 matrix, 3 vector
