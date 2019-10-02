from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


# Add more class labels as needed, make sure to start at 1
def class_text_to_int(row_label):
    if row_label == 'apple':
        return 1
    if row_label == 'banana':
        return 2
    if row_label == 'orange':
        return 3
    if row_label == 'beet':
        return 4
    if row_label == 'bread':
        return 5
    if row_label == 'butter':
        return 6
    if row_label == 'cabage':
        return 7
    if row_label == 'carrot':
        return 8
    if row_label == 'chillie':
        return 9
    if row_label == 'chillie powder':
        return 10
    if row_label == 'coconut':
        return 11
    if row_label == 'cookies':
        return 12
    if row_label == 'corn':
        return 13
    if row_label == 'fish':
        return 14
    if row_label == 'flour':
        return 15
    if row_label == 'grapes':
        return 16
    if row_label == 'green beans':
        return 17
    if row_label == 'guava':
        return 18
    if row_label == 'mango':
        return 19
    if row_label == 'meat':
        return 20
    if row_label == 'milk':
        return 21
    if row_label == 'oil':
        return 22
    if row_label == 'papaya':
        return 23
    if row_label == 'pineapple':
        return 24
    if row_label == 'pumpkin':
        return 25
    if row_label == 'rice':
        return 26
    else:
        None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    for i in ['test', 'train']:
        print('Iterating ' + i);
        writer = tf.python_io.TFRecordWriter(i+'.record')
        path = os.path.join(os.getcwd(), 'images/'+i)
        examples = pd.read_csv('data/'+i+'.csv')
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the '+i+ ' TFRecords')


if __name__ == '__main__':
    tf.app.run()
