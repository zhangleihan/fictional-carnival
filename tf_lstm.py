
import os
import numpy as np
import random
import dataloader
import tensorflow as tf
import tensorflow.contrib.layers as layers

map_fn = tf.map_fn

session = tf.Session()

train_x,train_y,val_x,val_y = dataloader.data_loader()

################################################################################
##                           GRAPH DEFINITION                                 ##
################################################################################

INPUT_SIZE    = 1      # 2 bits per timestep
RNN_HIDDEN    = 30
OUTPUT_SIZE   = 1      # 1 bit per timestep
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.001

USE_LSTM = True

inputs  = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)


## Here cell can be any function you want, provided it has two attributes:
#     - cell.zero_state(batch_size, dtype)- t state in __call__
#     - cell.__call__(input, state) - function that given input and previous
#                                     state returns tuple (output, state) where
#                                     state is the state passed to the next
#                                     timestep and output is the tensor used
#                                     for infering the output at timestep. For
#                                     example for LSTM, output is just hidden,
#                                     but state is memory + hidden
# Example LSTM cell with learnable zero_state can be found here:
#    https://gist.github.com/nivwusquorum/160d5cf7e1e82c21fad3ebf04f039317
if USE_LSTM:
    cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
else:
    cell = tf.contrib.rnn.BasicRNNCell(RNN_HIDDEN)

# Create initial state. Here it is just a constant tensor filled with zeros,
# but in principle it could be a learnable parameter. This is a bit tricky
# to do for LSTM's tuple state, but can be achieved by creating two vector
# Variables, which are then tiled along batch dimension and grouped into tuple.
batch_size    = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)

# Given inputs (time, batch, input_size) outputs a tuple
#  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
#  - states:  (time, batch, hidden_size)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)

# project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
# an extra layer here.
final_projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)

# apply projection to every timestep.
predicted_outputs = map_fn(final_projection, rnn_outputs)


# compute elementwise cross entropy.
#loss = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
#loss = tf.reduce_mean(loss)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predicted_outputs, labels=outputs))

#loss = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
#loss = tf.reduce_mean(loss)

# optimize
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
#train_fn = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# assuming that absolute difference between output and correct answer is 0.5
# or less we can round it to the correct output.
#accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))
correct_pred = tf.equal(tf.argmax(predicted_outputs,1), tf.argmax(outputs,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

################################################################################
##                           TRAINING LOOP                                    ##
################################################################################

NUM_BITS = 17
DIM1 = 30
DIM2 = 30
ITERATIONS_PER_EPOCH = 50
BATCH_SIZE = 64
COUNT1 = 0

#valid_x, valid_y, COUNT1 = dataloader.generate_batch(DIM1, DIM2, 2000, val_x, val_y, 0, NUM_BITS)

valid_x, valid_y, COUNT1 = dataloader.generate_batch(DIM1, DIM2, 2000, val_x, val_y, 1000, NUM_BITS)


COUNT2 = 0

# For some reason it is our job to do this:
session.run(tf.global_variables_initializer())

for epoch in range(45):
    iter_loss = 0
    epoch_loss = 0
    for i in range(ITERATIONS_PER_EPOCH):
        # here train_fn is what triggers backprop. loss and accuracy on their
        # own do not trigger the backprop.
        x, y, COUNT2 = dataloader.generate_batch(DIM1, DIM2, BATCH_SIZE, train_x, train_y, COUNT2, NUM_BITS)
        '''
        if epoch >= 0:
            print(session.run(predicted_outputs,feed_dict={inputs: x}))
            print COUNT2
            print('\n')
            #print(y)
            print(session.run(correct_pred,feed_dict={inputs: x, outputs: y}))
            raw_input("Press Enter to continue...")
        '''
        iter_loss = session.run([loss, train_fn], {
            inputs: x,
            outputs: y,
        })[0]
        iter_accuracy = session.run(accuracy, {inputs: x, outputs: y})
        print "Iter %d, train iter loss: %.2f, iter valid accuracy: %.1f %%" % (i, iter_loss, iter_accuracy * 100)
        epoch_loss += iter_loss

    epoch_loss /= ITERATIONS_PER_EPOCH

    valid_accuracy = session.run(accuracy, {
        #valid_x, valid_y, COUNT2 = dataloader.generate_batch(DIM1, DIM2, COUNT2, val_x, val_y, 0, NUM_BITS)
        inputs:  valid_x,
        outputs: valid_y,
    })
    print "Epoch %d, train loss: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_loss, valid_accuracy * 100.0)
    #print "Epoch %d, train loss: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_loss, accuracy.eval(session=session,feed_dict={inputs: valid_x, outputs: valid_y}))
    #print "Epoch %d, train loss: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_loss, accuracy.eval(session=session,feed_dict={inputs: x, outputs: y}))

