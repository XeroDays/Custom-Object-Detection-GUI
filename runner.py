from detect import *
from edit_model import *

#Bounding Box Object Detection Runner Script
#--------------------------------------------------------------------------------------
#using the model url passed form TensorFlow Object Detection Model Zoo
#Any image can be used but currently only objects already trained on can be detected 
#         for example a car can be but a type of cell phone cannot
# Use detect.downloadmodel to download new models from tensorflow model zoo. But be sure 
# you are in the correct directory before downloading
#
#

modelurl = 'http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz'


detect = detector()
#detect.readclasses(classfile)

detect.downloadmodel(modelurl)

#detect.loadModel()
#configpath, trainpath, testpath, lablepath, modelpath
#open_configs(configpath, trainpath, testpath, lablepath, modelpath)

#detect.predictimage(imagepath)