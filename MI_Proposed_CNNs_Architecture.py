#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Hide the Configuration and Warning
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# Import the Used Packages: Numpy, Pandas, and Tensorflow
import pandas as pd
import numpy as np
import tensorflow as tf

# Clear the Stack
tf.reset_default_graph()

# The Location of train_data，train_labels，test_data，test_labels
# DataSet Address
DIR = 'K:/Google_Driver/EEG_Features_For_Multi_class_Motor_Imagery/EEG_Test_Raw_Data/Changed_Excel_Data/'

# Model Saver Address
SAVE = 'K:/Google_Driver/EEG_Features_For_Multi_class_Motor_Imagery/EEG_Test_Raw_Data/First_Try_Model/'

# Activate a Session
sess = tf.InteractiveSession()

# Read Training Data
train_data = pd.read_csv(DIR + 'training_label.csv', header=None)
train_data = np.array(train_data).astype('float32')

# Read Training Labels
train_labels = pd.read_csv(DIR + 'Training_labels.csv', header=None)
train_labels = np.array(train_labels)

# Read Testing Data
test_data = pd.read_csv(DIR + 'Test_data.csv', header=None)
test_data = np.array(test_data).astype('float32')

# Read Testing Labels
test_labels = pd.read_csv(DIR + 'Test_labels.csv', header=None)
test_labels = np.array(test_labels)

# Set Batch Size 64
batch_size = 64
n_batch = train_data.shape[0] // batch_size

# Initialize the Weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

# Initialize the Bias
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

# Define the Function of Summary
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)

        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# Define Placeholders
with tf.name_scope("Input"):
    # x is the input feature data
    with tf.name_scope("Input_Data"):
        x = tf.placeholder(tf.float32, [None, 640])

    # y is the label related to the data
    with tf.name_scope("Labels"):
        y = tf.placeholder(tf.float32, [None, 4])

    # Keep_Prob is the possibility that keep neural while using dropout
    with tf.name_scope("Keep_Prob"):
        keep_prob = tf.placeholder(tf.float32)

    # Reshape the input data into 2-dimensional
    with tf.name_scope("Reshape_Data"):
        x_Reshape = tf.reshape(tensor=x, shape=[-1, 32, 20, 1])

# First Convolutional Layer
with tf.name_scope('Convolutional_1'):
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([3, 3, 1, 32])
        # variable_summaries(W_conv1)

    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32])
        # variable_summaries(b_conv1)

    with tf.name_scope('h_conv1'):
        h_conv1 = tf.nn.conv2d(x_Reshape, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        # variable_summaries(h_conv1)

    with tf.name_scope('h_conv1_Acti'):
        h_conv1_Acti = tf.nn.leaky_relu(h_conv1)
        # variable_summaries(h_conv1_Acti)

    with tf.name_scope('h_conv1_drop'):
        h_conv1_drop = tf.nn.dropout(h_conv1_Acti, keep_prob, noise_shape=[tf.shape(h_conv1_Acti)[0], 1, 1, tf.shape(h_conv1_Acti)[3]])
        # variable_summaries(h_conv1_drop)

# Second Convolutional Layer
with tf.name_scope('Convolutional_2'):
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([3, 3, 32, 32])
        # variable_summaries(W_conv2)

    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([32])
        # variable_summaries(b_conv2)

    with tf.name_scope('h_conv2'):
        h_conv2 = tf.nn.conv2d(h_conv1_drop, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
        # variable_summaries(h_conv2)

    with tf.name_scope('h_conv2_BN'):
        h_conv2_BN = tf.layers.batch_normalization(h_conv2, training=True)
        # variable_summaries(h_conv2_BN)

    with tf.name_scope('h_conv2_Acti'):
        h_conv2_Acti = tf.nn.leaky_relu(h_conv2_BN)
        # variable_summaries(h_conv2_Acti)

# Third Convolutional Layer
with tf.name_scope('Convolutional_3'):
    with tf.name_scope('W_conv3'):
        W_conv3 = weight_variable([3, 3, 64, 64])
        # variable_summaries(W_conv3)

    with tf.name_scope('b_conv3'):
        b_conv3 = bias_variable([64])
        # variable_summaries(b_conv3)

    with tf.name_scope('h_conv3_res'):
        h_conv3_res = tf.concat([h_conv2_Acti, h_conv1_drop], axis=3)
        # variable_summaries(h_conv3_res)

    with tf.name_scope('h_conv3'):
        h_conv3 = tf.nn.conv2d(h_conv3_res, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3
        # variable_summaries(h_conv3)

    with tf.name_scope('h_conv3_Acti'):
        h_conv3_Acti = tf.nn.leaky_relu(h_conv3)
        # variable_summaries(h_conv3_Acti)

    with tf.name_scope('h_pool3_drop'):
        h_conv3_drop = tf.nn.dropout(h_conv3_Acti, keep_prob, noise_shape=[tf.shape(h_conv3_Acti)[0], 1, 1, tf.shape(h_conv3_Acti)[3]])
        # variable_summaries(h_conv3_drop)

# First Max Pooling Layer
with tf.name_scope('Pooling_1'):
    with tf.name_scope('h_pool3'):
        h_pool3 = tf.nn.max_pool(h_conv3_drop, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # variable_summaries(h_pool3)

# Fourth Convolutional Layer
with tf.name_scope('Convolutional_4'):
    with tf.name_scope('W_conv4'):
        W_conv4 = weight_variable([3, 3, 64, 64])
        # variable_summaries(W_conv4)

    with tf.name_scope('b_conv4'):
        b_conv4 = bias_variable([64])
        # variable_summaries(b_conv4)

    with tf.name_scope('h_conv4'):
        h_conv4 = tf.nn.conv2d(h_pool3, W_conv4, strides=[1, 1, 1, 1], padding='VALID') + b_conv4
        # variable_summaries(h_conv4)

    with tf.name_scope('h_conv4_BN'):
        h_conv4_BN = tf.layers.batch_normalization(h_conv4, training=True)
        # variable_summaries(h_conv4_BN)

    with tf.name_scope('h_conv4_Acti'):
        h_conv4_Acti = tf.nn.leaky_relu(h_conv4_BN)
        # variable_summaries(h_conv4_Acti)

    with tf.name_scope('h_conv4_drop'):
        h_conv4_drop = tf.nn.dropout(h_conv4_Acti, keep_prob, noise_shape=[tf.shape(h_conv4_Acti)[0], 1, 1, tf.shape(h_conv4_Acti)[3]])
        # variable_summaries(h_conv4_drop)

# Fifth Convolutional Layer
with tf.name_scope('Convolutional_5'):
    with tf.name_scope('W_conv5'):
        W_conv5 = weight_variable([3, 3, 64, 64])
        # variable_summaries(W_conv5)

    with tf.name_scope('b_conv5'):
        b_conv5 = bias_variable([64])
        # variable_summaries(b_conv5)

    with tf.name_scope('h_conv5'):
        h_conv5 = tf.nn.conv2d(h_conv4_drop, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5
        # variable_summaries(h_conv5)

    with tf.name_scope('h_conv5_BN'):
        h_conv5_BN = tf.layers.batch_normalization(h_conv5, training=True)
        # variable_summaries(h_conv5_BN)

    with tf.name_scope('h_conv5_Acti'):
        h_conv5_Acti = tf.nn.leaky_relu(h_conv5_BN)
        # variable_summaries(h_conv5_Acti)

# Sixth Convolutional Layer
with tf.name_scope('Convolutional_6'):
    with tf.name_scope('W_conv6'):
        W_conv6 = weight_variable([3, 3, 128, 128])
        # variable_summaries(W_conv6)

    with tf.name_scope('b_conv6'):
        b_conv6 = bias_variable([128])
        # variable_summaries(b_conv6)

    with tf.name_scope('h_conv6_res'):
        h_conv6_res = tf.concat([h_conv5_Acti, h_conv4_drop], axis=3)
        # variable_summaries(h_conv6_res)

    with tf.name_scope('h_conv6'):
        h_conv6 = tf.nn.conv2d(h_conv6_res, W_conv6, strides=[1, 1, 1, 1], padding='SAME') + b_conv6
        # variable_summaries(h_conv6)

    with tf.name_scope('h_conv6_Activation'):
        h_conv6_Acti = tf.nn.leaky_relu(h_conv6)
        # variable_summaries(h_conv6_Acti)

    with tf.name_scope('h_pool6_drop'):
        h_conv6_drop = tf.nn.dropout(h_conv6_Acti, keep_prob, noise_shape=[tf.shape(h_conv6_Acti)[0], 1, 1, tf.shape(h_conv6_Acti)[3]])
        # variable_summaries(h_conv6_drop)

# Second Max Pooling Layer
with tf.name_scope('Pooling_2'):
    with tf.name_scope('h_pool6'):
        h_pool6 = tf.nn.max_pool(h_conv6_drop, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # variable_summaries(h_pool6)

# Flatten Layer
with tf.name_scope('Flatten'):
    with tf.name_scope('h_pool6_flat'):
        h_pool6_flat = tf.reshape(h_pool6, [-1, 4 * 7 * 128])
        # variable_summaries(h_pool6_flat)

# First Fully Connected Layer
with tf.name_scope('Fully_Connected_1'):
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([4 * 7 * 128, 512])
        # variable_summaries(W_fc1)

    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([512])
        # variable_summaries(b_fc1)

    with tf.name_scope('h_fc1'):
        h_fc1 = tf.matmul(h_pool6_flat, W_fc1) + b_fc1
        # variable_summaries(h_fc1)

    with tf.name_scope('h_fc1_BN'):
        h_fc1_BN = tf.layers.batch_normalization(h_fc1, training=True)
        # variable_summaries(h_fc1_BN)

    with tf.name_scope('h_fc1_Acti'):
        h_fc1_Acti = tf.nn.leaky_relu(h_fc1_BN)
        # variable_summaries(h_fc1_Acti)

    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1_Acti, keep_prob)
        # variable_summaries(h_fc1_drop)

# Second Fully Connected Layer
with tf.name_scope('Output_Layer'):
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([512, 4])
        # variable_summaries(W_fc2)

    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([4])
        # variable_summaries(b_fc2)

    with tf.name_scope('prediction'):
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        # variable_summaries(prediction)

# Define Loss Function
with tf.name_scope('loss'):
    with tf.name_scope('Euclidean_Distance'):
        loss = tf.reduce_mean(tf.square(y - prediction))
        tf.summary.scalar('loss', loss)

# Define Training Optimizer
with tf.name_scope('Train_Optimizer'):
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

# Calculate Accuracy
# Add metrics to TensorBoard.
with tf.name_scope('Evalution'):
    # Calculate Each Task Accuracy
    with tf.name_scope('Each_Class_accuracy'):
        # Task 1 Accuracy
        with tf.name_scope('T1_accuracy'):
            # Number of Classified Correctly
            y_T1 = tf.equal(tf.argmax(y, 1), 0)
            prediction_T1 = tf.equal(tf.argmax(prediction, 1), 0)
            T1_Corrected_Num = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T1, prediction_T1), tf.float32))

            # Number of All the Test Samples
            T1_all_Num = tf.reduce_sum(tf.cast(y_T1, tf.float32))

            # Task 1 Accuracy
            T1_accuracy = tf.divide(T1_Corrected_Num, T1_all_Num)
            tf.summary.scalar('T1_accuracy', T1_accuracy)

            T1_TP = T1_Corrected_Num
            T1_TN = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T1), tf.math.logical_not(prediction_T1)), tf.float32))
            T1_FP = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T1), prediction_T1), tf.float32))
            T1_FN = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T1, tf.math.logical_not(prediction_T1)), tf.float32))

            with tf.name_scope("T1_Precision"):
                T1_Precision = T1_TP / (T1_TP + T1_FP)
                tf.summary.scalar('T1_Precision', T1_Precision)

            with tf.name_scope("T1_Recall"):
                T1_Recall = T1_TP / (T1_TP + T1_FN)
                tf.summary.scalar('T1_Recall', T1_Recall)

            with tf.name_scope("T1_F_Score"):
                T1_F_Score = (2*T1_Precision*T1_Recall)/(T1_Precision+T1_Recall)
                tf.summary.scalar('T1_F_Score', T1_F_Score)

        # Task 2 Accuracy
        with tf.name_scope('T2_accuracy'):
            # Number of Classified Correctly
            y_T2 = tf.equal(tf.argmax(y, 1), 1)
            prediction_T2 = tf.equal(tf.argmax(prediction, 1), 1)
            T2_Corrected_Num = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T2, prediction_T2), tf.float32))

            # Number of All the Test Samples
            T2_all_Num = tf.reduce_sum(tf.cast(y_T2, tf.float32))

            # Task 2 Accuracy
            T2_accuracy = tf.divide(T2_Corrected_Num, T2_all_Num)
            tf.summary.scalar('T2_accuracy', T2_accuracy)

            T2_TP = T2_Corrected_Num
            T2_TN = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T2), tf.math.logical_not(prediction_T2)), tf.float32))
            T2_FP = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T2), prediction_T2), tf.float32))
            T2_FN = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T2, tf.math.logical_not(prediction_T2)), tf.float32))

            with tf.name_scope("T2_Precision"):
                T2_Precision = T2_TP / (T2_TP + T2_FP)
                tf.summary.scalar('T2_Precision', T2_Precision)

            with tf.name_scope("T2_Recall"):
                T2_Recall = T2_TP / (T2_TP + T2_FN)
                tf.summary.scalar('T2_Recall', T2_Recall)

            with tf.name_scope("T2_F_Score"):
                T2_F_Score = (2*T2_Precision*T2_Recall)/(T2_Precision+T2_Recall)
                tf.summary.scalar('T2_F_Score', T2_F_Score)

        # Task 3 Accuracy
        with tf.name_scope('T3_accuracy'):
            # Number of Classified Correctly
            y_T3 = tf.equal(tf.argmax(y, 1), 2)
            prediction_T3 = tf.equal(tf.argmax(prediction, 1), 2)
            T3_Corrected_Num = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T3, prediction_T3), tf.float32))

            # Number of All the Test Samples
            T3_all_Num = tf.reduce_sum(tf.cast(y_T3, tf.float32))

            # Task 3 Accuracy
            T3_accuracy = tf.divide(T3_Corrected_Num, T3_all_Num)
            tf.summary.scalar('T3_accuracy', T3_accuracy)

            T3_TP = T3_Corrected_Num
            T3_TN = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T3), tf.math.logical_not(prediction_T3)), tf.float32))
            T3_FP = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T3), prediction_T3), tf.float32))
            T3_FN = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T3, tf.math.logical_not(prediction_T3)), tf.float32))

            with tf.name_scope("T3_Precision"):
                T3_Precision = T3_TP / (T3_TP + T3_FP)
                tf.summary.scalar('T3_Precision', T3_Precision)

            with tf.name_scope("T3_Recall"):
                T3_Recall = T3_TP / (T3_TP + T3_FN)
                tf.summary.scalar('T3_Recall', T3_Recall)

            with tf.name_scope("T3_F_Score"):
                T3_F_Score = (2*T3_Precision*T3_Recall)/(T3_Precision+T3_Recall)
                tf.summary.scalar('T3_F_Score', T3_F_Score)

        # Task 4 Accuracy
        with tf.name_scope('T4_accuracy'):
            # Number of Classified Correctly
            y_T4 = tf.equal(tf.argmax(y, 1), 3)
            prediction_T4 = tf.equal(tf.argmax(prediction, 1), 3)
            T4_Corrected_Num = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T4, prediction_T4), tf.float32))

            # Number of All the Test Samples
            T4_all_Num = tf.reduce_sum(tf.cast(y_T4, tf.float32))

            # Task 4 Accuracy
            T4_accuracy = tf.divide(T4_Corrected_Num, T4_all_Num)
            tf.summary.scalar('T4_accuracy', T4_accuracy)

            T4_TP = T4_Corrected_Num
            T4_TN = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T4), tf.math.logical_not(prediction_T4)), tf.float32))
            T4_FP = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T4), prediction_T4), tf.float32))
            T4_FN = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T4, tf.math.logical_not(prediction_T4)), tf.float32))

            with tf.name_scope("T4_Precision"):
                T4_Precision = T4_TP / (T4_TP + T4_FP)
                tf.summary.scalar('T4_Precision', T4_Precision)

            with tf.name_scope("T4_Recall"):
                T4_Recall = T4_TP / (T4_TP + T4_FN)
                tf.summary.scalar('T4_Recall', T4_Recall)

            with tf.name_scope("T4_F_Score"):
                T4_F_Score = (2*T4_Precision*T4_Recall)/(T4_Precision+T4_Recall)
                tf.summary.scalar('T4_F_Score', T4_F_Score)

    # Calculate the Confusion Matrix
    with tf.name_scope("Confusion_Matrix"):
        with tf.name_scope("T1_Label"):
            T1_T1 = T1_Corrected_Num
            T1_T2 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T1, prediction_T2), tf.float32))
            T1_T3 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T1, prediction_T3), tf.float32))
            T1_T4 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T1, prediction_T4), tf.float32))

            T1_T1_percent = tf.divide(T1_T1, T1_all_Num)
            T1_T2_percent = tf.divide(T1_T2, T1_all_Num)
            T1_T3_percent = tf.divide(T1_T3, T1_all_Num)
            T1_T4_percent = tf.divide(T1_T4, T1_all_Num)

            tf.summary.scalar('T1_T1_percent', T1_T1_percent)
            tf.summary.scalar('T1_T2_percent', T1_T2_percent)
            tf.summary.scalar('T1_T3_percent', T1_T3_percent)
            tf.summary.scalar('T1_T4_percent', T1_T4_percent)

        with tf.name_scope("T2_Label"):
            T2_T1 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T2, prediction_T1), tf.float32))
            T2_T2 = T2_Corrected_Num
            T2_T3 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T2, prediction_T3), tf.float32))
            T2_T4 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T2, prediction_T4), tf.float32))

            T2_T1_percent = tf.divide(T2_T1, T2_all_Num)
            T2_T2_percent = tf.divide(T2_T2, T2_all_Num)
            T2_T3_percent = tf.divide(T2_T3, T2_all_Num)
            T2_T4_percent = tf.divide(T2_T4, T2_all_Num)

            tf.summary.scalar('T2_T1_percent', T2_T1_percent)
            tf.summary.scalar('T2_T2_percent', T2_T2_percent)
            tf.summary.scalar('T2_T3_percent', T2_T3_percent)
            tf.summary.scalar('T2_T4_percent', T2_T4_percent)

        with tf.name_scope("T3_Label"):
            T3_T1 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T3, prediction_T1), tf.float32))
            T3_T2 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T3, prediction_T2), tf.float32))
            T3_T3 = T3_Corrected_Num
            T3_T4 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T3, prediction_T4), tf.float32))

            T3_T1_percent = tf.divide(T3_T1, T3_all_Num)
            T3_T2_percent = tf.divide(T3_T2, T3_all_Num)
            T3_T3_percent = tf.divide(T3_T3, T3_all_Num)
            T3_T4_percent = tf.divide(T3_T4, T3_all_Num)

            tf.summary.scalar('T3_T1_percent', T3_T1_percent)
            tf.summary.scalar('T3_T2_percent', T3_T2_percent)
            tf.summary.scalar('T3_T3_percent', T3_T3_percent)
            tf.summary.scalar('T3_T4_percent', T3_T4_percent)

        with tf.name_scope("T4_Label"):
            T4_T1 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T4, prediction_T1), tf.float32))
            T4_T2 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T4, prediction_T2), tf.float32))
            T4_T3 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T4, prediction_T3), tf.float32))
            T4_T4 = T4_Corrected_Num

            T4_T1_percent = tf.divide(T4_T1, T4_all_Num)
            T4_T2_percent = tf.divide(T4_T2, T4_all_Num)
            T4_T3_percent = tf.divide(T4_T3, T4_all_Num)
            T4_T4_percent = tf.divide(T4_T4, T4_all_Num)

            tf.summary.scalar('T4_T1_percent', T4_T1_percent)
            tf.summary.scalar('T4_T2_percent', T4_T2_percent)
            tf.summary.scalar('T4_T3_percent', T4_T3_percent)
            tf.summary.scalar('T4_T4_percent', T4_T4_percent)

    with tf.name_scope('Global_Evalution_Metrics'):
        # Global Average Accuracy - Simple Algorithm
        with tf.name_scope('Global_Average_Accuracy'):
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            Global_Average_Accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('Global_Average_Accuracy', Global_Average_Accuracy)

        with tf.name_scope('Kappa_Metric'):
            Test_Set_Num = T1_all_Num + T2_all_Num + T3_all_Num + T4_all_Num

            Actual_T1 = T1_all_Num
            Actual_T2 = T2_all_Num
            Actual_T3 = T3_all_Num
            Actual_T4 = T4_all_Num

            Prediction_T1 = T1_T1 + T2_T1 + T3_T1 + T4_T1
            Prediction_T2 = T1_T2 + T2_T2 + T3_T2 + T4_T2
            Prediction_T3 = T1_T3 + T2_T3 + T3_T3 + T4_T3
            Prediction_T4 = T1_T4 + T2_T4 + T3_T4 + T4_T4

            p0 = (T1_T1 + T2_T2 + T3_T3 + T4_T4) / Test_Set_Num
            pe = (Actual_T1*Prediction_T1 + Actual_T2*Prediction_T2 + Actual_T3*Prediction_T3 + Actual_T4*Prediction_T4) / \
                 (Test_Set_Num*Test_Set_Num)

            Kappa_Metric = (p0 - pe) / (1 - pe)
            tf.summary.scalar('Kappa_Metric', Kappa_Metric)

        with tf.name_scope('Micro_Averaged_Evalution'):
            with tf.name_scope("Micro_Averaged_Confusion_Matrix"):
                TP_all = T1_TP + T2_TP + T3_TP + T4_TP
                TN_all = T1_TN + T2_TN + T3_TN + T4_TN
                FP_all = T1_FP + T2_FP + T3_FP + T4_FP
                FN_all = T1_FN + T2_FN + T3_FN + T4_FN

            with tf.name_scope("Micro_Global_Precision"):
                Micro_Global_Precision = TP_all / (TP_all + FP_all)
                tf.summary.scalar('Micro_Global_Precision', Micro_Global_Precision)

            with tf.name_scope("Micro_Global_Recall"):
                Micro_Global_Recall = TP_all / (TP_all + FN_all)
                tf.summary.scalar('Micro_Global_Recall', Micro_Global_Recall)

            with tf.name_scope("Micro_Global_F1_Score"):
                Micro_Global_F1_Score = (2*Micro_Global_Precision*Micro_Global_Recall)/(Micro_Global_Precision+Micro_Global_Recall)
                tf.summary.scalar('Micro_Global_F1_Score', Micro_Global_F1_Score)

        with tf.name_scope('Macro_Averaged_Evalution'):
            with tf.name_scope("Macro_Global_Precision"):
                Macro_Global_Precision = (T1_Precision + T2_Precision + T3_Precision + T4_Precision) / 4
                tf.summary.scalar('Macro_Global_Precision', Macro_Global_Precision)

            with tf.name_scope("Macro_Global_Recall"):
                Macro_Global_Recall = (T1_Recall + T2_Recall + T3_Recall + T4_Recall) / 4
                tf.summary.scalar('Macro_Global_Recall', Macro_Global_Recall)

            with tf.name_scope("Macro_Global_F1_Score"):
                Macro_Global_F1_Score = (T1_F_Score + T2_F_Score + T3_F_Score + T4_F_Score) / 4
                tf.summary.scalar('Macro_Global_F1_Score', Macro_Global_F1_Score)

# Merge all the summaries
merged = tf.summary.merge_all()

# Initialize all the variables
sess.run(tf.global_variables_initializer())

# Start a saver to save the trained model
saver = tf.train.Saver()

# Summary the Training and Test Processing
train_writer = tf.summary.FileWriter(SAVE + 'train_Writer', sess.graph)
test_writer  = tf.summary.FileWriter(SAVE + 'test_Writer')

for epoch in range(2019):
    for batch_index in range(n_batch):
        random_batch = random.sample(range(train_data.shape[0]), batch_size)
        batch_xs = train_data[random_batch]
        batch_ys = train_labels[random_batch]
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.50})

    # Accuracy on Training Set
    train_acc, train_loss = sess.run([Global_Average_Accuracy, loss], feed_dict={x: train_data, y: train_labels, keep_prob: 1.0})

    # Accuracy on Test Set
    test_summary, test_acc, test_loss = sess.run([merged, Global_Average_Accuracy, loss], feed_dict={x: test_data, y: test_labels, keep_prob: 1.0})
    test_writer.add_summary(test_summary, epoch)
    
    # Show the Model Capability
    print("Iter " + str(epoch) + ", Training Accuracy: " + str(train_acc) + ", Testing Accuracy: " + str(test_acc))

    # Save the Model Every 100 Epoches
    if epoch % 100 == 0:
        saver.save(sess, save_path=SAVE + 'Model_Saver/Ite_%s' % epoch)

    if epoch == 2001:
        output_prediction = sess.run(prediction, feed_dict={x: test_data, y: test_labels, keep_prob: 1.0})
        np.savetxt(SAVE + "prediction.csv", output_prediction, delimiter=",")
        np.savetxt(SAVE + "labels.csv", test_labels, delimiter=",")

train_writer.close()
test_writer.close()
sess.close()
