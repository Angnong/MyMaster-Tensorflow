import csv
import os
import sys
import os.path
import argparse

import numpy as np
import tensorflow as tf

import logging
import re
import json

import multiprocessing


def parse_args():
    parser = argparse.ArgumentParser(description='Convert folder or file with annotation files to TFRecoders.')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='verbose terminal output')
    parser.add_argument('-i', '--input', dest='input', required=True, type=str, help='input path')
    parser.add_argument('-p', '--image_path', dest='search_path', type=str, help='image search path')
    parser.add_argument('-o', '--output', dest='output', required=True, type=str, help='output path')
    parser.add_argument('-c', '--class_id', dest='class_id', type=str, help='class id assignment')
    parser.add_argument('-f', '--filter', dest='filter', type=str, help='filter for the input files')
    parser.add_argument(
        '-t',
        '--type',
        dest='model',
        type=str,
        default='voc',
        choices=['voc', 'caffe', 'own'],
        help='type definition of class labels')
    parser.add_argument('-s', '--shards', dest='shards', type=int, default=128, help='count of shards')

    parser.add_argument('-b', '--bbox', dest='bbox', type=str, help='bounding box file')

    args = parser.parse_args()
    return args


'''
'''


def create_example(image_raw, label_values, label_names, filename, bbox):
    # convert to bytes
    label_names = [x.encode('utf-8') for x in label_names]

    return tf.train.Example(features=tf.train.Features(feature={
        'image/label/values':
        tf.train.Feature(int64_list=tf.train.Int64List(value=label_values)),
        'image/label/names':
        tf.train.Feature(bytes_list=tf.train.BytesList(value=label_names)),
        'image/filename':
        tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
        'image/encoded':
        tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
        'image/bbox/xmin':
        tf.train.Feature(float_list=tf.train.FloatList(value=[bbox[0]])),
        'image/bbox/ymin':
        tf.train.Feature(float_list=tf.train.FloatList(value=[bbox[1]])),
        'image/bbox/width':
        tf.train.Feature(float_list=tf.train.FloatList(value=[bbox[2]])),
        'image/bbox/height':
        tf.train.Feature(float_list=tf.train.FloatList(value=[bbox[3]])),
    }))


'''
image_label_list:
    filename -> label_list
label_id_map:
    label -> index(label)
'''


def convert_data_batch(image_label_list, label_id_map, image_bbox_list, out_path, ranges, thread_id, shards):
    session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.01)))

    jpeg_data = tf.placeholder(dtype=tf.string)

    jpeg_decode = tf.image.decode_jpeg(jpeg_data, channels=3)

    num_shards_per_batch = shards // len(ranges)

    shard_ranges = np.linspace(ranges[thread_id][0], ranges[thread_id][1], num_shards_per_batch + 1).astype(int)

    for x in range(num_shards_per_batch):

        shard = thread_id * num_shards_per_batch + x
        record_path = os.path.join(out_path, '{}_{}.rec'.format(shard, shards - 1))
        with tf.python_io.TFRecordWriter(record_path) as writer:

            for i in range(shard_ranges[x], shard_ranges[x + 1]):
                image_name = image_label_list[i][0]
                image_path = image_label_list[i][0]
                bbox = image_bbox_list[i][1]
                if image_path is None:
                    logging.warning('{} not found in {}'.format(image_name, image_path))
                    continue
                # read file
                with tf.gfile.FastGFile(image_path, 'r') as f:
                    image_data = f.read()

                try:
                    var = session.run(jpeg_decode, {jpeg_data: image_data})
                except tf.python.errors.InvalidArgumentError:
                    logging.warning('{} not jpeg'.format(image_name))
                    continue

                # write line
                label_names = image_label_list[i][1]
                label_values = [label_id_map[x] for x in label_names]

                writer.write(
                    create_example(image_data, label_values, label_names, image_name, bbox).SerializeToString())
                logging.info('write {} to record'.format(image_name))


def convert_data(image_label_list, label_id_map, image_bbox_list, out_path, threads=8, shards=1024):

    # create output path
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # create_ranges
    spacing = np.linspace(0, len(image_label_list), threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    process_list = []
    for x in range(threads):
        process = multiprocessing.Process(
            target=convert_data_batch,
            args=[image_label_list, label_id_map, image_bbox_list, out_path, ranges, x, shards])
        process.start()
        process_list.append(process)

    for x in process_list:
        x.join()

    logging.info('write finished')


'''
TODO change this function
create two maps:
    images_with_labels -> map with filepath : label
    class_id_map -> map with label : label_index
'''


def convert_dir(in_path, out_path, bbox_path, shards=128, file_filter=None, model='voc', class_id_file=None):
    logging.info('start conversion')

    # angnong: change code from here to suit CASIA database
    # just extract the face region
    # start from here
    images_with_labels = {}
    images_with_bbox = {}
    class_id_map = {}
    cnt_images = 0
    max_label = 0
    # read from csv file
    with open(in_path, 'r') as f:
        content = csv.reader(f, delimiter=',')
        for line in content:
            if len(line) < 2:
                continue

            file_name = line[0]
            label_value = int(line[1])
            if label_value > max_label:
                max_label = label_value

            folder_path = os.path.dirname(file_name)
            label_name = os.path.basename(folder_path)
            img_name = os.path.splitext(os.path.basename(file_name))[0]

            img_bbox_path = bbox_path + os.path.sep + label_name + os.path.sep + img_name + '.csv'

            if not os.path.isfile(img_bbox_path):
                print('Cannot find csv: ' + img_bbox_path)
                continue

            images_with_labels[file_name] = [label_name]

            with open(img_bbox_path, 'r') as bbox_file:
                content = csv.reader(bbox_file, delimiter=';')
                for line in content:
                    if len(line) < 4:
                        print('Empty file: ' + img_bbox_path)
                        continue

                    x = int(line[0])
                    y = int(line[1])
                    w = int(line[2])
                    h = int(line[3])

                    if x == -1 or y == -1 or w == -1 or h == -1:
                        print('Invalid value in file: ' + img_bbox_path)
                        continue

                    bbox_value = [x, y, w, h]

            cnt_images += 1
            images_with_bbox[file_name] = bbox_value
            class_id_map[label_name] = label_value

    # write json file with informations about the dataset
    data_for_json = {
        'num_images': cnt_images,
        'num_classes': max_label + 1,
        'multi_label': False,
        'dir_path': '.',
        'file_list': []
    }
    with open(out_path + '/' + 'dataset.json', 'w') as f:
        json.dump(data_for_json, f, sort_keys=False, indent=2)

    # convert images_with_labels in a list
    image_label_list = [(k, v) for (k, v) in images_with_labels.items()]
    # convert images with bbox in a list
    image_bbox_list = [(m, n) for (m, n) in images_with_bbox.items()]
    convert_data(image_label_list, class_id_map, image_bbox_list, out_path, shards=shards)


def main():
    args = parse_args()

    #bbox_path = '/home/zhanga/master/database/CASIA/faces_all_0.95'
    level = logging.ERROR
    if args.verbose:
        level = logging.DEBUG

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=level)

    if not os.path.exists(args.input):
        logging.error('file or dir not found')
        return -1

    if os.path.isfile(args.input):
        convert_dir(
            args.input,
            args.output,
            args.bbox,
            #bbox_path,
            file_filter=args.filter,
            shards=args.shards,
            model=args.model,
            class_id_file=args.class_id)

    return 0


if __name__ == '__main__':
    sys.exit(main())
