# python imports
import datetime
import json
import numpy as np
import os
from shutil import copyfile
import sys
import tensorflow as tf
import argparse
import tensorflow.contrib.slim as slim

import cnn_architectures

from dataset import Dataset
import image_processing
import logging
'''
########################################################################################################################
FUNCTIONS
########################################################################################################################
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Training a CNN classifier')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-c', '--config', dest='config_path', type=str, required=True, help='path to the config file')
    args = parser.parse_args()
    return args


#
# def loss(logits, labels):
#     '''Args:
#     logits: Logits from inference().
#     labels: Labels of the input image pf batch size
#
#     Returns:
#     Loss tensor of type float.'''
#
#     # first the average cross entropy loss across the batch
#     labels = tf.cast(labels, tf.int64)  # if needed,change to type int64
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
#     cross_entropy_mean = tf.reduce_mean(cross_entropy)
#     #tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_mean)
#     # The total loss is defined as the cross entropy loss plus all the weight
#     # decay terms (L2 loss).
#     return tf.add_n([cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='total_loss')
#


def build_train_graph(config, dataset):

    with tf.device('/cpu:0'):
        inputs, labels = image_processing.distorted_inputs(
            dataset,
            batch_size=config['parameters']['batch_size'],
            height=config['input']['height'],
            width=config['input']['width'],
            channels=config['input']['channels'],
            add_variations=config['parameters']['additional_variations'],
            num_preprocess_threads=8)

    with tf.device('/gpu:0'):
        logits, endpoints = cnn_architectures.create_model(
            config['model']['architecture'],
            inputs,
            is_training=True,
            num_classes=config['input']['classes'],
            reuse=None)

    labels = tf.cast(labels, tf.int64)  # if needed,change to type int64
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = tf.add_n([cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='total_loss')

    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('train/accuracy', accuracy, collections=['train'])
    tf.summary.scalar('train/loss', loss, collections=['train'])

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var, collections=['train'])

    return loss, accuracy, tf.summary.merge_all(key='train')


def build_val_graph(config, dataset):

    with tf.device('/cpu:0'):
        inputs, labels = image_processing.inputs(
            dataset,
            batch_size=config['parameters']['batch_size'],
            height=config['input']['height'],
            width=config['input']['width'],
            channels=config['input']['channels'],
            num_preprocess_threads=8)

    with tf.device('/gpu:0'):
        logits, endpoints = cnn_architectures.create_model(
            config['model']['architecture'],
            inputs,
            is_training=False,
            num_classes=config['input']['classes'],
            reuse=True)

    labels = tf.cast(labels, tf.int64)  # if needed,change to type int64
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = tf.add_n([cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='total_loss')

    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('val/accuracy', accuracy, collections=['validation'])
    tf.summary.scalar('val/loss', loss, collections=['validation'])

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var, collections=['validation'])

    return loss, accuracy, tf.summary.merge_all(key='validation')


def train(config):

    train_dataset = Dataset(os.path.join(config['input']['path'], 'train', 'dataset.json'))
    val_dataset = Dataset(os.path.join(config['input']['path'], 'val', 'dataset.json'))

    with tf.Graph().as_default():
        with tf.name_scope('train') as scope:
            train_loss, train_accuracy, train_summary = build_train_graph(config, train_dataset)

        with tf.name_scope('val') as scope:
            val_loss, val_accuracy, val_summary = build_val_graph(config, val_dataset)
        # print(dir(tf.GraphKeys))
        # print(tf.get_collection(tf.GraphKeys.LOSSES))
        # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     print(x.name)

        if config['model']['finetune']:
            exclude = cnn_architectures.model_weight_excludes(config['model']['architecture'])
            variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(config['model']['checkpoint'],
                                                                         variables_to_restore)

            # set learning rate
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(config['parameters']['base_lr'], global_step,
                                                   config['parameters']['step_size'], config['parameters']['gamma'])

        # minimize losses
        train_step = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=config['parameters']['momentum']).minimize(
                train_loss, global_step=global_step)

        # batch operation
        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if batchnorm_updates:
            batchnorm_updates = tf.group(*batchnorm_updates)

        # Create a saver
        saver = tf.train.Saver(max_to_keep=config['output']['keep_last_k_models'])

        # initialize
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if config['model']['finetune']:
                sess.run(init_assign_op, init_feed_dict)

            # run summary writer
            summary_writer = tf.summary.FileWriter(config['output']['path'], sess.graph)

            # Start the queue runners.
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)
            logging.info('graph built')

            for step in range(config['parameters']['iterations']):
                if step % config['parameters']['validation_iter'] == 0:
                    com_acc = 0.0
                    com_loss = 0.0
                    count = 0
                    for x in range(val_dataset.num_images() // config['parameters']['batch_size'] + 1):
                        acc_v, loss_v, step_v, summary_v = sess.run([val_accuracy, val_loss, global_step, val_summary])
                        com_acc += acc_v
                        com_loss += loss_v
                        count += 1

                    logging.info('val: step {}, accuracy = {:.2f}, loss = {:.2f}'.format(step_v, com_acc / count,
                                                                                         com_loss / count))

                    summary_writer.add_summary(summary_v, step_v)
                # train with summaries
                if step % config['output']['display_results'] == 0:

                    acc_v, _, loss_v, learn_v, _, step_v, summary_v = sess.run([
                        train_accuracy, train_step, train_loss, learning_rate, batchnorm_updates, global_step,
                        train_summary
                    ])

                    logging.info('train: step {}, accuracy = {:.2f}, loss = {:.2f}, lr = {:.5f}'.format(
                        step_v - 1, acc_v, loss_v, learn_v))

                    summary_writer.add_summary(summary_v, step_v - 1)

                else:
                    acc_v, _, loss_v, learn_v, _, step_v = sess.run(
                        [train_accuracy, train_step, train_loss, learning_rate, batchnorm_updates, global_step])

                # save the model
                if step_v % config['output']['save_iterations'] == 0:
                    checkpoint_path = os.path.join(config['output']['path'], 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step_v)
                    logging.info('Model saved in file: {}'.format(checkpoint_path))

            coord.request_stop()
            coord.join()


'''
########################################################################################################################
MAIN
########################################################################################################################
'''


def main(argv=None):
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.ERROR
    if args.verbose:
        level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s ', datefmt='%d-%m-%Y %H:%M:%S', level=level)

    # load config
    with open(args.config_path) as config_file:
        config = json.load(config_file)

    # setup output folder
    if tf.gfile.Exists(config['output']['path']):
        while True:
            sys.stdout.write('Output path already exists. Do you want to overwrite? [Y/n]: ')
            choice = input().lower()
            if choice == '' or choice == 'y':
                tf.gfile.DeleteRecursively(config['output']['path'])
                break
            elif choice == 'n':
                sys.exit(0)

    tf.gfile.MakeDirs(config['output']['path'])

    # save used config file to output folder
    copyfile(args.config_path, os.path.join(config['output']['path'], 'config.json'))

    # start training
    train(config)


if __name__ == '__main__':
    tf.app.run()
