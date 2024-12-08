from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
from random import shuffle
import numpy as np
import logging
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

logger = logging.getLogger("Main")
logger.setLevel(logging.DEBUG)


'''
Group of functions to move annotation .txt file to tensorflow TFRECORDS file type
'''
    
def class_text_to_int(row_label, lablemap):
    '''
    Used to give each class type a numerical value
    '''
    with open(lablemap, "r") as file:
        lines = file.readlines()

        for c,i in enumerate(lines):
            text = i.replace("\n", "").replace("\'","").split(":")

            if text[len(text)-1] == row_label:
                output = lines[c-1].replace("\n", "").replace("\'","").split(":")
                return int(output[len(output)-1])
        print(f"ERROR: No lable named {row_label} found")    
        logger.error(f"ERROR: No lable named {row_label} found")
        return None


def splitit(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, lablepath):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
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
        classes.append(class_text_to_int(row['class'], lablepath))

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


def generatecsv(csv, resultpath, imgpath, lablepath):
    '''
    Main fuction to create tfrecords
    
    args:
        cvs,             pandas dataframe
        result_path,     FILE PATH pointing to models .h5 file
        image_path,      FOLDER PATH containing all images to be used for both training and testing
    '''
    
    
    writer = tf.io.TFRecordWriter(resultpath)
    path = os.path.join(imgpath)

    
    grouped = splitit(csv, 'filename')
        
    for group in grouped:
        tf_example = create_tf_example(group, path, lablepath)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), resultpath)
    print('Successfully created the TFRecords: {}'.format(output_path))