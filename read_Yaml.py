import yaml
import os

'''
Yaml file loader. Used to parse a given yaml file for use with Model_Train.
Yaml file should look like:
---
model_Name: "TestFRCNN"
image_Path: 'Pictures/FolderContainingImages'
model_Path: 'Documents/models/TestFRCNN'
config_Path: 'Documents/models/TestFRCNN/pipeline.config'
test_Path: 'Documents/models/'
train_Path: 'Documents/models/'
label_Path: 'Documents/models/Label_Map_Full_Setter.pbtxt'
api_Path: 'Path to installed Tensorflow/Models folder'
checkpoint_Path: 'Documents/models/TestFRCNN/checkpoint0'
results_Path: 'Documents/models/results.h5'
annotations_Path: 'Documents/models/AnnotationsAll.txt'
test_Record: 'Documents/models/test.record'
train_Record: 'Documents/models/train.record'
controlled: 1 #(If using seperate test and train annotation files set to 1)
base_Resolution: [3088,2064] #(image width, image height)
final_Resolution: 1333 #(resolution of images for training)
model_Type: 1 #(see model train for help)
setup: [2,3,0.001,1500000] #Setup is: Number of classes, batch size, learning rate, steps
test_annotations: 'Documents/models/AnnotsAllTest.txt'
train_annotations: 'Documents/models/AnnotsAllTrain.txt'

'''

def load_yaml(yaml_file):
    stream = open(yaml_file, 'r')
    dic = yaml.safe_load(stream)
    out = []
    for key, value in dic.items():
        if key == 'model_Name':
            out.append(value)
            continue
        elif key == 'controlled':
            out.append(int(value))
            continue
        elif key == 'base_Resolution':
            out.append(value)
            continue
        elif key == 'final_Resolution':
            out.append(int(value))
            continue
        elif key == 'model_Type':
            out.append(int(value))
            continue
        elif key == 'setup':
            out.append(value)
            continue
        
        path = os.path.join(os.path.expanduser('~'), str(value))
        out.append(path)
    return out