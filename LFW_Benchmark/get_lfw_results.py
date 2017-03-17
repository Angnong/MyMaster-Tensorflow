import argparse
import csv
import json
import math
import numpy as np
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise
import tensorflow as tf
import tensorflow.contrib.slim as slim

sys.path.insert(1, '/home/zhanga/master/TensorFlow/CNN_Training')
from cnn_architectures import inception_resnet_v2
from cnn_architectures import inception_utils
from cnn_architectures import inception_v4
from cnn_architectures import resnet_utils
from cnn_architectures import resnet_v1
from cnn_architectures import resnet_v2
from cnn_architectures import scratchnet
from cnn_architectures import vgg
from dataset import Dataset
'''
########################################################################################################################
FUNCTIONS
########################################################################################################################
'''


def parse_args():
    parser = argparse.ArgumentParser(description='LFW Evaluation')
    parser.add_argument('-l', '--lfw', dest='LFW_path', type=str, required=True, help='Path to LFW')
    parser.add_argument('-p1', '--prefix1', dest='prefix_img1', type=str, required=True, help='Prefix of first image')
    parser.add_argument('-p2', '--prefix2', dest='prefix_img2', type=str, required=True, help='Prefix of second image')
    parser.add_argument('-m', '--model', dest='model_file', type=str, required=True, help='Path to the trained model')
    args = parser.parse_args()
    return args


def init_cnn(sess, args, config, images_placeholder):
    architectures = {
        'inception_resnet_v2': {
            'model': inception_resnet_v2.inception_resnet_v2,
            'layer': 'FOO'
        },
        'inception_v4': {
            'model': inception_v4.inception_v4,
            'layer': 'FOO'
        },
        'resnet_v1_50': {
            'model': resnet_v1.resnet_v1_50,
            'scope': resnet_v1.resnet_arg_scope,
            'layer': 'FOO',
            'reduce_mean': True
        },
        'resnet_v1_101': {
            'model': resnet_v1.resnet_v1_101,
            'scope': resnet_v1.resnet_arg_scope,
            'layer': 'resnet_v1_101/block4/unit_3/bottleneck_v1',
            'reduce_mean': True
        },
        'resnet_v1_152': {
            'model': resnet_v1.resnet_v1_152,
            'scope': resnet_v1.resnet_arg_scope,
            'layer': 'resnet_v1_152/block4/unit_3/bottleneck_v1',
            'reduce_mean': True
        },
        'scratchnet': {
            'model': scratchnet.scratchnet,
            'layer': 'scratchnet/pool5',
            'reduce_mean': False
        },
        'vgg_16': {
            'model': vgg.vgg_16,
            'layer': 'vgg_16/pool5',
            'reduce_mean': False
        },
        'vgg_19': {
            'model': vgg.vgg_19,
            'layer': 'vgg_19/pool5',
            'reduce_mean': False
        }
    }

    if config['model']['architecture'] not in architectures:
        print('CNN model not found')
        return -1

    model = architectures[config['model']['architecture']]['model']

    if 'scope' in architectures[config['model']['architecture']]:
        scope = architectures[config['model']['architecture']]['scope']
        with slim.arg_scope(scope()):
            logits, endpoints = model(images_placeholder, num_classes=config['input']['classes'], is_training=False)
    else:
        logits, endpoints = model(images_placeholder, num_classes=config['input']['classes'], is_training=False)

    #print(endpoints)

    net = endpoints[architectures[config['model']['architecture']]['layer']]

    saver = tf.train.Saver()
    saver.restore(sess, args.model_file)
    print('---------------------------')
    print('Restore model from: {}'.format(args.model_file))

    if architectures[config['model']['architecture']]['reduce_mean'] == True:
        net = tf.reduce_mean(net, [1, 2], name='last_pool', keep_dims=True)

    net = tf.squeeze(net)
    return net


def img_preprocess(img_encode, config):
    # decode the image
    img = tf.image.decode_jpeg(img_encode)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # normalize image
    img = tf.image.crop_to_bounding_box(img, 50, 63, 150, 125)  # TODO: ATTENTION
    img = tf.expand_dims(img, 0)
    img = tf.image.resize_bilinear(img, [config['input']['height'], config['input']['width']])
    if config['input']['channels'] == 1:
        img = tf.image.rgb_to_grayscale(img)

    # Finally, rescale to [-1,1] instead of [0, 1)
    img = tf.subtract(img, 0.5)
    img = tf.multiply(img, 2.0)
    return img


def load_image(sess, args, img_preprocess, img_encode, config, image_path):
    with tf.gfile.FastGFile(os.path.join(args.LFW_path, image_path), 'r') as f:
        img = f.read()

    return sess.run(img_preprocess, feed_dict={img_encode: img})  # convert tensor to np.array


def get_deep_representations(sess, args, config, image_placeholder, endpoints, img_paths):
    print('Calculate deep representations of LFW images ... ')

    cur_imgs = np.zeros((config['parameters']['batch_size'], config['input']['width'], config['input']['height'],
                         config['input']['channels']))
    cur_img_paths = []
    deep_representations = {}
    batch_iter = 0

    img_encode = tf.placeholder(dtype=tf.string)
    img_pre = img_preprocess(img_encode, config)

    for i in range(len(img_paths)):
        img = load_image(sess, args, img_pre, img_encode, config, img_paths[i])
        cur_imgs[batch_iter, :, :, :] = img
        cur_img_paths.append(img_paths[i])

        # feed forward batch of images in cnn and extract feature vector
        batch_iter += 1
        if batch_iter == config['parameters']['batch_size']:
            x = sess.run(endpoints, feed_dict={image_placeholder: cur_imgs})
            for j in range(len(cur_img_paths)):
                deep_representations[cur_img_paths[j]] = x[j, :].reshape(1, -1)

            cur_img_paths = []
            batch_iter = 0

    x = sess.run(endpoints, feed_dict={image_placeholder: cur_imgs})
    for j in range(len(cur_img_paths)):
        deep_representations[cur_img_paths[j]] = x[j, :].reshape(1, -1)

    return deep_representations


def get_lfw_pairs(args):
    pairs = []
    labels = []
    img_paths = []
    with open(os.path.join(args.LFW_path + '/lfw', 'pairs.txt'), 'r') as f:
        content = csv.reader(f, delimiter='\t')
        for line in content:  # because the first line is not the pair
            if len(line) == 3:
                path_img1 = args.prefix_img1 + '/' + line[0] + '/' + line[0] + '_' + line[1].zfill(4) + '.jpg'
                path_img2 = args.prefix_img2 + '/' + line[0] + '/' + line[0] + '_' + line[2].zfill(4) + '.jpg'
                labels.append(1)
            elif len(line) == 4:
                path_img1 = args.prefix_img1 + '/' + line[0] + '/' + line[0] + '_' + line[1].zfill(4) + '.jpg'
                path_img2 = args.prefix_img2 + '/' + line[2] + '/' + line[2] + '_' + line[3].zfill(4) + '.jpg'
                labels.append(0)
            else:
                continue

            path_img1 = os.path.normpath(path_img1)
            path_img2 = os.path.normpath(path_img2)
            pairs.append([path_img1, path_img2])

            if path_img1 not in img_paths:
                img_paths.append(path_img1)
            if path_img2 not in img_paths:
                img_paths.append(path_img2)

    return np.asarray(pairs), img_paths, np.asarray(labels)


def get_distances(pairs, deep_representations):
    print('Computing distances for image pairs ...')

    distances = []
    count = 0
    for pair in pairs:
        x1 = deep_representations[pair[0]]
        x2 = deep_representations[pair[1]]
        dist = pairwise.cosine_similarity(x1, x2)[0, 0]
        distances.append(dist)

    return np.asarray(distances)


def find_best_thresh(labels, distances):
    bestThresh = -1
    bestThreshAcc = 0

    for thresh in distances:
        cnt_correct_predictions = 0
        for i in range(0, len(labels)):
            if distances[i] < thresh:
                y_pred = 0
            else:
                y_pred = 1

            if y_pred == labels[i]:
                cnt_correct_predictions += 1

        accuracy = cnt_correct_predictions * 1.0 / len(labels)
        if accuracy > bestThreshAcc:
            bestThresh = thresh
            bestThreshAcc = accuracy

    return bestThresh


def calc_fold_accuracy(pairs, labels, distances, thresh):
    cnt_correct_predictions = 0
    falseDetections = []
    for i in range(0, len(labels)):
        if distances[i] < thresh:
            y_pred = 0
        else:
            y_pred = 1

        if y_pred == labels[i]:
            cnt_correct_predictions += 1
        else:
            falseDetections.append([pairs[i][0], pairs[i][1], labels[i], distances[i]])

    accuracy = float(cnt_correct_predictions) / len(labels)
    return accuracy, falseDetections


def main():
    args = parse_args()
    print(args)

    # load config
    with open(os.path.join(os.path.dirname(args.model_file), 'config.json')) as config_file:
        config = json.load(config_file)

    # load pairs
    pairs, img_paths, labels = get_lfw_pairs(args)

    # setup result path
    result_path = os.path.join(os.path.dirname(args.model_file), 'results_LFW')
    result_prefix = os.path.basename(args.model_file)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # init tf session
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        image_placeholder = tf.placeholder(
            tf.float32,
            shape=(config['parameters']['batch_size'], config['input']['height'], config['input']['width'],
                   config['input']['channels']))
        endpoints = init_cnn(sess, args, config, image_placeholder)
        deep_representations = get_deep_representations(sess, args, config, image_placeholder, endpoints, img_paths)

    # get distances
    distances = get_distances(pairs, deep_representations)

    # k fold cross validation
    print('Perform cross validation ... ')
    folds = KFold(n_splits=10, shuffle=False, random_state=None)
    accuracies = []
    thresholds = []
    falseDetections = []

    with open(os.path.join(result_path, result_prefix + '_' + args.prefix_img1 + '_' +
    args.prefix_img2 + '_results.csv'), 'w') as LFW_result_file:
        writer = csv.writer(LFW_result_file)
        for idx, (train, test) in enumerate(folds.split(distances)):
            print('\tSplit ' + str(idx + 1))
            thresh = find_best_thresh(labels[train], distances[train])
            accuracy, fold_falseDetections = calc_fold_accuracy(pairs[test], labels[test], distances[test], thresh)

            accuracies.append(accuracy)
            thresholds.append(thresh)
            falseDetections.append(fold_falseDetections)
            writer.writerow(['fold: %d, accuracy: %f, threshold: %f' % (idx, accuracy, thresh)])

        accuracies = np.asarray(accuracies)
        thresholds = np.asarray(thresholds)

        writer.writerow(['---------------------------'])
        writer.writerow(['average_accuracy: %f, standard_deviation: %f' % (np.mean(accuracies), np.std(accuracies))])
        writer.writerow(['average_threshold: %f, standard_deviation: %f' % (np.mean(thresholds), np.std(thresholds))])

    # write false predicted pairs into csv
    with open(os.path.join(result_path, result_prefix + '_' + args.prefix_img1 + '_' +
        args.prefix_img2 + '_falsePairs.csv'), 'w') as LFW_falseDetections_file:
        writer = csv.writer(LFW_falseDetections_file, delimiter=',')
        for fold_falseDetections in falseDetections:
            for falseDetection in fold_falseDetections:
                writer.writerow(falseDetection)

    # write the average accuracy and std of diff combinations to one file for compariation            
    with open(os.path.join(result_path,'merge_all_accuracy.csv'),'a') as LFW_merge:
        writer = csv.writer(LFW_merge, delimiter=',')
        writer.writerow([result_prefix + '_' + args.prefix_img1 + '_' + args.prefix_img2 , 
            'average_accuracy:' , np.mean(accuracies), 'standard_deviation:', 
            np.std(accuracies)])
    
    # TODO: Draw ROC


if __name__ == '__main__':
    main()
