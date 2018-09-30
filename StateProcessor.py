#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:35:17 2018

@author: rgarzon
"""
import tensorflow as tf

class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    def __init__(self,n_input):
        # Build the Tensorflow graph
        self.max_input_value = int (n_input /2)
        self.min_input_value = 0
        with tf.variable_scope("state_processor"):
            #self.input_state = tf.placeholder(tf.float32)   
            self.input_state = tf.placeholder(shape=[n_input], dtype=tf.float32)
            self.output = self.input_state        
            #self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            #self.output = tf.image.rgb_to_grayscale(self.input_state)
            #self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            #self.output = tf.image.resize_images(
            #    self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            #self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })


    def process_with_normalization(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        #print (state)
        current_min = self.min_input_value
        current_max = self.max_input_value
        state_norm = (state - ((current_max-current_min)/2)) / (current_max - ((current_max-current_min)/2))
        #print (state_norm)
        return sess.run(self.output, { self.input_state: state_norm })