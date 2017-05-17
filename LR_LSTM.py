
import os
import sys
import numpy as np
import dataloader

import tensorflow as tf


class LogisticRegression(object):
    def __init__(self):
        self.X = tf.placeholder("float", (None, None, 1))
        self.Y = tf.placeholder("float", (None, None, 1))

        self.W = tf.Variable(tf.random_normal((28 * 28, 10,1), stddev=0.01))
        self.b = tf.Variable(tf.zeros([10, ]))

        self.model = self.create_model(self.X, self.W, self.b)

        # logistic and cal error
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.Y))

        # gradient descent method to minimize error
        self.train = tf.train.GradientDescentOptimizer(0.1).minimize(self.cost)
        # calculate the max pos each row
        self.predict = tf.argmax(self.model, 1)


    def create_model(self, X, w, b):
        # wx + b
        return tf.add(tf.matmul(X, w), b)


    def run(self):
        train_x,train_y,val_x,val_y = dataloader.data_loader()
        #train_x, train_y = train_set
        #val_x, val_y = test_set
        #train_y = self.dense_to_one_hot(train_y)
        #val_y = self.dense_to_one_hot(val_y)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(100):
            for start, end in zip(range(0, len(train_x), 128), range(128, len(train_x), 128)):
                sess.run(self.train, feed_dict={self.X: train_x[start:end], self.Y: train_y[start:end]})
            print i, np.mean(np.argmax(val_y, axis=1) == sess.run(self.predict, feed_dict={self.X: val_x, self.Y: val_y}))

        sess.close()


lr_model = LogisticRegression()
lr_model.run()