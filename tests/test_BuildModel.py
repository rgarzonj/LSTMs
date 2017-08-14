#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:15:52 2017

@author: rubengarzon
"""
import tensorflow as tf
import os
import sys
sys.path.append("/Users/rubengarzon/Documents/Projects/phD/Repo/gym")
import gym
import numpy as np
import random


    
numBlocks = 5
n_steps = 3
n_input = 10
n_hidden = 32

class StateProcessor():
    """
    Processes a raw Atari iamges. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(tf.float32)
            self.output = self.input_state

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """


    # RNN output node weights and biases

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
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
            'out': tf.Variable(tf.random_normal([n_hidden, numBlocks*numBlocks]))
            }
        biases = {
            'out': tf.Variable(tf.random_normal([numBlocks*numBlocks]))
            }
 
       # Batch size x time steps x features.
        # Batch size x Sequence Length (n_input)            
        self.X_pl = tf.placeholder(shape=[None,n_steps,n_input],dtype = tf.float32,name = "X")
        
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
                # reshape to [1, n_input]
        
        batch_size = tf.shape(self.X_pl)[0]
 #       x = tf.unstack(self.X_pl, n_steps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, self.X_pl, dtype=tf.float32)

        val = tf.transpose(outputs, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)

        # Linear activation, using rnn inner loop last output
        self.predictions = tf.matmul(last, weights['out']) + biases['out']
        #self.predictions = tf.matmul(outputs, weights['out']) + biases['out']
        
        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calcualte the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
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
        return sess.run(self.predictions, { self.X_pl: s })

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
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


# In[ ]:

# For Testing....
env = gym.envs.make("BlocksWorld-v0")

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)

e = Estimator(scope="test")
sp = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Example observation batch
    observation = env.reset()
    
    replay_memory = []
    
    for i in range(18):
        next_action = [random.randint(0,numBlocks-1),random.randint(0,numBlocks)] 
        observation, reward, done, empty = env.step(next_action)
        replay_memory.append(observation)        

    observations = np.array(replay_memory)
    observations = observations.reshape(-1,n_steps,n_input)
    # Test Prediction
    res = e.predict(sess, observations)
    print (res)
    print (res.shape)

    # Test training step
    y = np.array([10.0, 10.0, 10.0,10.0,10.0,10.0])
    a = np.array([1,3,5,7,9,11])
    print(e.update(sess, observations, a, y))
    
    
