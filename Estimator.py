#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 15:09:21 2018

@author: rgarzon
"""
import tensorflow as tf
import os

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", valid_actions=0, n_input=0,summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        self.VALID_ACTIONS = valid_actions
        self.n_input = n_input
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """
        
        weights = {
            'out': tf.Variable(tf.random_normal([len(self.VALID_ACTIONS), len(self.VALID_ACTIONS)]))
            }
        biases = {
            'out': tf.Variable(tf.random_normal([len(self.VALID_ACTIONS)]))
            }
 
        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.X_pl = tf.placeholder(shape=[None,self.n_input],dtype=tf.float32,name = "X")
        #self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        #X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        #conv1 = tf.contrib.layers.conv2d(
        #    X, 32, 8, 4, activation_fn=tf.nn.relu)
        #conv2 = tf.contrib.layers.conv2d(
        #    conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        #conv3 = tf.contrib.layers.conv2d(
        #    conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        #flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(self.X_pl, 32)
        fc2 = tf.contrib.layers.fully_connected(fc1, 32)
#        fc3 = tf.contrib.layers.fully_connected(fc2, 12)
        last = tf.contrib.layers.fully_connected(fc2, len(self.VALID_ACTIONS))
        
        # We need the network to output negative numbers (Rewards are negative or zero, so we add another final linear layer)
        self.predictions = tf.matmul(last, weights['out']) + biases['out']
        #print (self.predictions.shape)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calcualte the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
#        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
#        self.optimizer = tf.train.RMSPropOptimizer(0.0025, 0.99, 0.0, 1e-6)
        self.optimizer = tf.train.AdamOptimizer(0.0005)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        self.max_q_value = tf.reduce_max(self.predictions,1)
        #print (self.max_q_value.shape)
        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions)),
            tf.summary.histogram("fc1",fc1),
            tf.summary.histogram("fc2",fc2),
            tf.summary.histogram("last",last)
        ])

    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        #print ('State')
        #print (s)
        ret = sess.run(self.predictions, { self.X_pl: s })
        #print ('Actions')
        #print (ret)
        return ret
    
    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss, max_q_value = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss,self.max_q_value],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)        
        return loss,max_q_value
