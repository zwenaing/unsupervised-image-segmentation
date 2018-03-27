from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from os.path import exists
from encoder import encode
from decoder import decode
from input_data import input_data

logdir = "checkpoints/model.ckpt"

# network parameters
learning_rate = 0.0001
num_steps = 1000
display_step = 1

global_step = 0

X = tf.placeholder(tf.float32, [None, 224, 224, None])

with tf.name_scope("Encoding"):
    encoded_image = encode(X)

with tf.name_scope("Decoding"):
    decoded_image = decode(encoded_image)

with tf.name_scope("Loss"):
    y_pred = tf.reshape(decoded_image, [-1, 150528])
    y_true = tf.reshape(X, [-1, 150528])
    loss = tf.reduce_mean(tf.pow(y_pred - y_true, 2))
tf.summary.scalar("SEE loss", loss)

with tf.name_scope("Optimization"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss)

merged_summary = tf.summary.merge_all()

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    if exists(logdir):
        saver.restore(sess, logdir)
    iterator = input_data()

    for i in range(num_steps + 1):
        next_items = iterator.get_next()
        batch_x =  sess.run(next_items)
        _ = sess.run(train_op, feed_dict={X: batch_x})

        if i % display_step == 0:
            loss_, summary = sess.run([loss, merged_summary], feed_dict={X: batch_x})
            print("Iteration number: ", str(i), " Loss: ", str(loss_))
            train_writer.add_summary(summary)
            saver.save(sess, logdir, global_step=global_step + i)






