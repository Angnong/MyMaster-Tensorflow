# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:17:10 2017

@author: zhanga


"""

from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
import argparse
import os
import json
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import csv
import sys

sys.path.insert(1, '/home/zhanga/master/TensorFlow/CNN_Training')
from cnn_architectures import inception_resnet_v2
from cnn_architectures import inception_utils
from cnn_architectures import inception_v4
from cnn_architectures import resnet_utils
from cnn_architectures import resnet_v1
from cnn_architectures import resnet_v2
from cnn_architectures import scratchnet
from cnn_architectures import vgg
import cnn

##############################################
# para #
_select_percent = 0.5

###############################################
# Function #
###############################################


def parse_args():
    parser = argparse.ArgumentParser(description='Compare Deep Representations')
    parser.add_argument(
        '-p',
        '--pairlist',
        dest='list_path',
        type=str,
        required=True,
        help='Path to the list of image and correspoding variations')
    parser.add_argument(
        '-bbox', '--boundingbox', dest='bbox', type=str, required=True, help='Path to face bounding box file')
    parser.add_argument('-m', '--model', dest='model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument(
        '-i', '--image_folder', dest='imgdata_path', type=str, required=True, help='Path to image dataset')
    parser.add_argument(
        '-var', '--synthesized_list', dest='var_list', type=str, required=True, help='Path to only synthesized faces')
    args = parser.parse_args()
    return args


def init_cnn(sess, args, config, images_placeholder):
    '''
    some comment
    '''

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
            'layer': 'resnet_v1_50/block4/unit_3/bottleneck_v1',
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
    model = architectures[config['model']['architecture']]['model']

    if 'scope' in architectures[config['model']['architecture']]:
        scope = architectures[config['model']['architecture']]['scope']
        with slim.arg_scope(scope()):
            logits, endpoints = model(images_placeholder, num_classes=config['input']['classes'], is_training=False)
    else:
        logits, endpoints = model(images_placeholder, num_classes=config['input']['classes'], is_training=False)

    net = endpoints[architectures[config['model']['architecture']]['layer']]

    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)
    print('---------------------------')
    print('Restore model from: {}'.format(args.model_path))

    if architectures[config['model']['architecture']]['reduce_mean'] == True:
        net = tf.reduce_mean(net, [1, 2], name='last_pool', keep_dims=True)

    net = tf.squeeze(net)
    return net


def get_DR(sess, args, config, image_placeholder, endpoints, img_paths):
    """
    output: Deep representations
    """
    cur_imgs = np.zeros((config['parameters']['batch_size'], config['input']['width'], config['input']['height'],
                         config['input']['channels']))
    cur_img_paths = []
    deep_representations = []
    batch_iter = 0

    img_encode = tf.placeholder(dtype=tf.string)
    bbox = tf.placeholder(dtype=tf.int32)
    img_pre = img_preprocess(img_encode, config, bbox)

    for i in range(len(img_paths)):
        img = load_image(sess, args, img_pre, img_encode, bbox, config, img_paths[i][0])
        if img is None:
            #print('Invalid bounding box')
            continue

        cur_imgs[batch_iter, :, :, :] = img
        cur_img_paths.append(img_paths[i][0])

        # feed forward batch of images in cnn and extract feature vector
        batch_iter += 1
        if batch_iter == config['parameters']['batch_size']:
            x = sess.run(endpoints, feed_dict={image_placeholder: cur_imgs})
            for j in range(len(cur_img_paths)):
                deep_representations.append(x[j, :].reshape(1, -1))

            cur_img_paths = []
            batch_iter = 0

    x = sess.run(endpoints, feed_dict={image_placeholder: cur_imgs})
    for j in range(len(cur_img_paths)):
        deep_representations.append(x[j, :].reshape(1, -1))

    return deep_representations


def img_preprocess(img_encode, config, bbox):

    # decode the image
    img = tf.image.decode_jpeg(img_encode)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # crop with bounding box
    x, y, height, width = bbox[0], bbox[1], bbox[2], bbox[3]
    img = tf.image.crop_to_bounding_box(img, x, y, height, width)
    img = tf.expand_dims(img, 0)
    img = tf.image.resize_bilinear(img, [config['input']['height'], config['input']['width']])
    if config['input']['channels'] == 1:
        img = tf.image.rgb_to_grayscale(img)

    # Finally, rescale to [-1,1] instead of [0, 1)
    img = tf.subtract(img, 0.5)
    img = tf.multiply(img, 2.0)
    return img


def load_image(sess, args, img_preprocess, img_encode, bbox, config, img_path):

    # get bounding box
    class_name = os.path.basename(os.path.dirname(img_path))
    img_name = os.path.basename(img_path)
    with open(os.path.join(args.bbox, class_name, img_name[0:len(img_name) - 4] + '.csv'), 'r') as bbox_file:
        content1 = csv.reader(bbox_file, delimiter=';')
        for line in content1:
            if len(line) < 4:
                return None

            x = int(line[0])
            y = int(line[1])
            height = int(line[2])
            width = int(line[3])
            #print('bounding box:', x, y, height, width)
            bbox_value = [x, y, height, width]
            # get image data
            with tf.gfile.FastGFile(img_path, 'r') as f:
                img = f.read()
            return sess.run(img_preprocess, feed_dict={img_encode: img, bbox: bbox_value})


def get_var_pairs(list_path, bbox_path):
    """
    read from train list and seperate to pairs
    example variation pair:class/person [varpair1,varpair2,varpair3....]
                           all pairs [class1, class2, ....]
    and

    filter with bounding boxes. only select the ones with bounding box
    """
    class_wrap = []
    all_pairs = []
    #cont_line = 0
    img_idx_pre = 0
    int(img_idx_pre)
    with open(list_path, 'r') as image_list:
        content = csv.reader(image_list, delimiter='\t')
        cont_line = 0
        for line in content:
            img_path, img_idx = line[0].split(',')
            img_idx = int(img_idx)
            img_name = os.path.basename(img_path)
            class_name = os.path.basename(os.path.dirname(img_path))
            # check if it has bounding box
            if not os.path.exists(os.path.join(bbox_path, class_name, img_name[0:len(img_name) - 4] + '.csv')):
                cont_line += 1
                print('not find bounding box of', class_name + '/' + img_name, 'count:', cont_line)
                continue

            # wrap by class
            if img_idx == img_idx_pre:
                class_wrap.append([img_path, img_idx])
            else:
                all_pairs.append(class_wrap)
                class_wrap = []
                class_wrap.append([img_path, img_idx])
                img_idx_pre = img_idx
            """
            # wrap to 'pairs'
            if img_idx == img_idx_pre: # same class/person
                if cont_line % 7 == 0:
                    if len(pairs_wrap) != 0:
                        class_wrap.append(pairs_wrap)
                        pairs_wrap = [] # renew pair
                        pairs_wrap.append(os.path.join(class_name,img_name))
                        img_idx_pre = img_idx
                else:
                    pairs_wrap.append(os.path.join(class_name,img_name))
                    img_idx_pre = img_idx
            else:
                all_pairs.append(class_wrap)
                class_wrap = [] #renew class
                if cont_line % 7 == 0:
                    class_wrap.append(pairs_wrap)
                    pairs_wrap = [] # renew pair
                    pairs_wrap.append(os.path.join(class_name,img_name))
                    img_idx_pre = img_idx
                else:
                    pairs_wrap.append(os.path.join(class_name,img_name))
                    img_idx_pre = img_idx
            """

    print(len(all_pairs))
    return all_pairs


def get_var_similarity(avg_dr, var_pair_dr):
    # compare the vairations with class center and return the similarity/distances
    similarities = []
    for i in var_pair_dr:
        simi = pairwise.cosine_similarity(i.reshape(1, -1), avg_dr.reshape(1, -1))[0, 0]
        similarities.append(simi)

    return similarities


def re_select_vars_all_rank(class_var_list, select_thres):
    # rank all variation similarities and take the top select_thres percent in the whole ranking
    all_simi = []
    new_list = []
    for i in range(len(class_var_list)):
        all_simi.append(class_var_list[i][1])
    sort_idx = np.argsort(all_simi)
    select_num = int(select_thres * len(class_var_list))
    sort_idx_slc = sort_idx[len(class_var_list) - select_num:len(class_var_list)]
    for j in sort_idx_slc:
        new_list.append([class_var_list[j][0], all_simi[j]])

    print(len(new_list))

    return new_list


def re_select_vars(var_pair_dr, avg_class_dr, select_thres, class_var_pairs):
    # compare the variation with the class center and cut the threshold percent in the ranking in the class
    new_list = []
    similarities = []
    for i in var_pair_dr:
        simi = pairwise.cosine_similarity(i.reshape(1, -1), avg_class_dr.reshape(1, -1))[0, 0]
        similarities.append(simi)
    sort_idx = np.argsort(similarities)
    select_num = int(select_thres * len(similarities))
    sort_idx_slc = sort_idx[len(similarities) - select_num:len(similarities)]
    print(sort_idx_slc)
    for j in sort_idx_slc:
        new_list.append(class_var_pairs[j])

    return new_list


def get_class_attribute(deep_representations):
    avg_class_dr = np.mean(deep_representations, axis=0)
    # Criterion for re-selection
    # option 1: min_similarity: cosine similarity
    # option 2: max_distance: absolute distance
    min_similarity = 1
    max_distance = 0
    for dr in deep_representations:
        simi = pairwise.cosine_similarity(dr.reshape(1, -1), avg_class_dr.reshape(1, -1))[0, 0]
        if simi < min_similarity:
            min_similarity = simi
    for dr in deep_representations:
        dis = np.linalg.norm(dr - avg_class_dr)
        if dis > max_distance:
            max_distance = dis
    max_differ = [min_similarity, max_distance]
    return avg_class_dr[0], max_differ


def plot_class_avg_dr(avg_dr):
    class_to_draw = len(avg_dr)  # number of classes to draw
    color_choice = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    line_style = ['solid', 'dashed', 'dashdot', 'dotted']
    for i in range(class_to_draw):
        plt.plot(avg_dr[i], color_choice[i])
    plt.show()


def main():
    args = parse_args()
    print(args)

    # load config
    # config.json is under the current fold
    with open(os.path.join(os.getcwd(), 'config.json')) as config_file:
        config = json.load(config_file)

    # setup result path
    result_path = os.path.join(os.getcwd(), 'results_DR_compare')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    '''
    # check if already exits saved results
    file_path = os.path.join(result_path, 'avg_dr_by_class.csv')
    if os.path.exists(file_path):
        class_ground_truth = []
        with open(file_path, 'r') as avg_file:
            reader = csv.reader(avg_file, delimiter=',')
            for line in reader:
                class_ground_truth.append(line)
    '''
    # load pairs(original image and its pose variations) as a pair
    pairs = get_var_pairs(args.list_path, args.bbox)
    pairs_var = get_var_pairs(args.var_list, args.bbox)

    print(len(pairs), len(pairs_var))
    # input image pairs forward CNN #
    # init tf session
    with tf.Session() as sess:
        image_placeholder = tf.placeholder(
            tf.float32,
            shape=(config['parameters']['batch_size'], config['input']['height'], config['input']['width'],
                   config['input']['channels']))
        endpoints = init_cnn(sess, args, config, image_placeholder)

        # calculate step by step by class/person
        # save dictionary [class name] [class_avg_dr] [similarity and eculiden distance]
        class_dic = {}
        class_var_list = []
        for cont_idx, class_pairs in enumerate(pairs):
            print('calculate Deep Representations of original images by class: ', cont_idx, 'image number in class',
                  len(class_pairs))

            deep_representations = get_DR(sess, args, config, image_placeholder, endpoints, class_pairs)
            avg_class_dr, max_differ = get_class_attribute(deep_representations)
            with open(os.path.join(result_path, 'avg_dr_by_class.csv'), 'a') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([class_pairs[0][1], avg_class_dr, max_differ])
            class_dic[class_pairs[0][1]] = [avg_class_dr, max_differ]

        # calculate synthesized faces Deep Representations and select the good variations

        for class_var_idx, class_var_pairs in enumerate(pairs_var):
            # class_var_pairs = pairs_var[cont_idx]
            print('number of variations in class: ', class_var_idx, 'is: ', len(class_var_pairs))
            var_deep_representations = get_DR(sess, args, config, image_placeholder, endpoints, class_var_pairs)
            # find corresponding class_avg_dr in dictionary
            avg_class_dr = class_dic[class_var_pairs[0][1]][0]
            similarities = get_var_similarity(avg_class_dr, var_deep_representations)
            for item_idx, item_var in enumerate(class_var_pairs):
                class_var_list.append([item_var, similarities[item_idx]])

            # reselect_var_pairs = re_select_vars(var_deep_representations, avg_class_dr, 0.6, class_var_pairs)
        reselect_var_pairs = re_select_vars_all_rank(class_var_list, _select_percent)

        with open(os.path.join(result_path, 'reselect_trainlist_synthesize.csv'), 'a') as renew_pair:
            writer = csv.writer(renew_pair, delimiter=',')
            for item1 in reselect_var_pairs:
                writer.writerow(item1)
            '''
            # try visualize first 10 class avg dr as example
            # TODO: further change it to random 10 classes
            first10_avg_dr = []
            with open((os.path.join(result_path,'avg_dr_by_class.csv'),'r') as visual_class:
                dr_reader = csv.reader(visual_class,delimiter='/t')
                cont_class = 0
                for line in dr_reader:
                    if cont_class == 5:
                        break
                    else:
                        first5_avg_dr.append(line[1])
                        cont_class += 1
            # plot as curves
            plot_class_avg_dr(first5_avg_dr)
            '''


if __name__ == '__main__':
    main()
