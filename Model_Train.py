from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
import subprocess
import object_detection
import tensorflow as tf
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from detect import *
from generate_tfrecord import *
from edit_model import *
from tensorflow import keras
from shuffle_lines import shuffle_lines
import logging
import signal
from queue import Queue, Empty
import threading 
import csv
import fnmatch
import logging
import traceback

'''
This class is used to train tensorflow object detection models. 

args :  modelname,       STRING corrisponding to saved model folder name
        image_path,      FOLDER PATH containing all images to be used for both training and testing
        model_path,      FOLDER PATH to downloaded model folder
        config_path,     FILE PATH pointing to pipeline.config file
        test_path,       FOLDER PATH to location of test.record
        train_path,      FOLDER PATH to location of test.record
        lable_path,      FILE PATH pointing to lablemap in .pbtxt form
        apimodel_path,   FOLDER PATH to tensorflow object detection models folder
        checkpoint_path, FOLDER PATH to selected models checkpoint folder (rename this from defaut to checkpoint0)
        result_path,     FILE PATH pointing to models .h5 file
        annot_path,      FILE PATH pointing to annotations .txt file (use vgg annotator and convert .csv to .txt by changing the endswitch using rename DO NOT open in excel and save to .txt)
        test_record,     FILE PATH pointing to models test.record file
        train_record,    FILE PATH pointing to models train.record file 
        controlled,      Set to 1 if user needs controlled spilt of train and test data select 0 if random split is OK
        base_img_resolution, Resolution in pix of image used for annotations
        image_resolution, Model image resolution
        selector, 1 for fasterRcnn model or 2 for centernet model, 3 for SSD 4 for effdet

         
Help:
    Use detect.py to download model (make sure to be in the correct workspace when doing this)
    Use imge_edit.py to covert all images to consistant width and height
    Use edit_model.py to edit model pipeline.config file
    Use TF_Detect.py to evaluate the trained model

'''

#Setup Logging to objdet.log file
# create logger with 'spam_application'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('logTraining.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

#If not set tensorflow will fail due to Autograph Errors
# Since we are subprocessing a seperate CX_Freexe executable the
# matrix values cannot be passed using tensorflows autograph.
tf.config.run_functions_eagerly(True)
tf.autograph.set_verbosity(0)

class model_trainer():
    def __init__(self, modelname, image_path, model_path, 
                 config_path, test_path, train_path, 
                 label_path, apimodel_path, checkpoint_path,
                 result_path, annot_path, test_record,
                 train_record, controlled, base_img_resolution_width, 
                 base_img_resolution_height, image_resolution, selector) -> None:
        
        self.modelname     = modelname
        self.imagepth      = image_path
        self.modelpth      = model_path
        self.configpth     = config_path
        self.testpth       = test_path
        self.trainpth      = train_path
        self.lablepth      = label_path
        self.apimodelpth   = apimodel_path
        self.checkpointpth = checkpoint_path
        self.resultpth     = result_path
        self.annotpth      = annot_path
        self.test_record   = test_record
        self.train_record  = train_record
        self.controlled    = controlled
        self.base_Wres     = base_img_resolution_width
        self.base_Hres     = base_img_resolution_height
        self.image_res     = image_resolution
        self.selector      = selector
        
        self.Error_During_Training = False

        self.scalefactorX = self.image_res/self.base_Wres
        self.scalefactorY = self.image_res/self.base_Hres
        
        self.data = []
        self.targets = []
        self.filenames = []
            
        self.pipe_edit = pipeline_editor()
        self.pipeline_filled_flag = False
        
        self.dataframe = dict()
        self.dataframe["filename"] = list()
        self.dataframe["width"] = list()
        self.dataframe["height"] = list()
        self.dataframe["xmin"] = list()
        self.dataframe["xmax"] = list()
        self.dataframe["ymin"] = list()
        self.dataframe["ymax"] = list()
        self.dataframe["class"] = list()
    
        
        if controlled:
            self.data_test = []
            self.targets_test = []
            self.filenames_test = []
            
            self.dataframe_test = dict()
            self.dataframe_test["filename"] = list()
            self.dataframe_test["width"] = list()
            self.dataframe_test["height"] = list()
            self.dataframe_test["xmin"] = list()
            self.dataframe_test["xmax"] = list()
            self.dataframe_test["ymin"] = list()
            self.dataframe_test["ymax"] = list()
            self.dataframe_test["class"] = list()
        
        
    def uncontrolled_split(self):
        '''
        Uncontrolled split for random data split for test and train data
        '''
        
        # Annotation lines are shuffled to ensure all classes are treated equally
        
        # Finding Annotations file in dir
        for root, dir, files in os.walk(self.annotpth):
            for basename in files:
                if fnmatch.fnmatch(basename, '*.csv'):
                    self.annotations = os.path.join(root, basename).replace("\\","/")
                    ext = '.csv'
                elif fnmatch.fnmatch(basename, '*.txt'):
                    self.annotations = os.path.join(root, basename).replace("\\","/")
                    ext = '.txt'
        

        shuffle_lines(self.annotations)        

        if ext == ".csv":
            lines = open(self.annotations, newline='')
            rows = csv.reader(lines,delimiter=',' )
        
            for row in rows:
                #parsing .txt file to obtain key values 

                #Catching lines without data
                try:
                    val = int(row[1])
                except ValueError:
                    continue
                  
    
                (filename, size, delim, rc ,ri,vals, type) = row
                vals = vals.split(":")
            
                xx = int(vals[2].strip("\"':,y")) * self.scalefactorX
                yy = int(vals[3].strip("\"':,width")) * self.scalefactorY
                ww = int(vals[4].strip("\"':,height")) * self.scalefactorX
                hh = int(vals[5].strip("\"':,}")) * self.scalefactorY
                type = type.strip("\"':{}")

                if (xx < 0) or (yy< 0) or (ww< 0) or (hh < 0):
                    print("error bounding box has negative value")
                    break
    
                ####################BAD DATA CHECK#############################
                '''
                tensorflow expects boundig boxes to be created from topleft to bottom right
                if data does not follow this error will be thrown
                np.nim and np.max solve this issue programically 
                '''
                # Data from labling doesnt give xmax but gives width, so to get 
                # Xmax we add xmin and the width together
            
                # NOTE different image annotation software will give different methods of 
                # Bounding box dimensions. Use the defined annotation software (https://annotate.officialstatistics.org/)
                # to avoid reprogramming. 
                xmin_new = xx
                xmax_new = xx + ww
                ymin_new = yy
                ymax_new = yy + hh
            
                if (xmax_new > self.image_res) or (xmin_new > self.image_res) or (ymax_new > self.image_res) or (ymax_new > self.image_res):
                    continue
    
                ##################END BAD DATA CHECK##########################

                fullpath = os.path.sep.join([self.imagepth, filename]).replace("\\", "/")
                self.dataframe["filename"].append(fullpath)

                image = cv2.imread(fullpath)
                (height, width) = image.shape[:2]

                self.dataframe["width"].append(width)
                self.dataframe["height"].append(height)

                self.dataframe["class"].append(type)

                self.dataframe["xmin"].append(xmin_new)
                self.dataframe["ymin"].append(ymin_new)
                self.dataframe["xmax"].append(xmax_new )
                self.dataframe["ymax"].append(ymax_new)
                
                
                
        elif ext == ".txt":
            rows = open(self.annotations).read().strip().split("\n")
        
            for row in rows:
                #parsing .txt file to obtain key values 
                row=row.split(",")
                
                try:
                    tst = int(row[1])
                except ValueError:
                    continue
    
                (filename, size, delim, rc ,ri,shape, x, y, w, h, type) = row
                xx = int(x.strip("\"':x")) * self.scalefactorX
                yy = int(y.strip("\"':y")) * self.scalefactorY
                ww = int(w.strip("\"':width")) * self.scalefactorX
                hh = int(h.strip("\"':height}")) * self.scalefactorY
                type = type.strip("\"':{}")

                if (xx < 0) or (yy< 0) or (ww< 0) or (hh < 0):
                    print("error bounding box has negative value")
                    break
    
                ####################BAD DATA CHECK#############################
                '''
                tensorflow expects boundig boxes to be created from topleft to bottom right
                if data does not follow this error will be thrown
                np.nim and np.max solve this issue programically 
                '''
                # Data from labling doesnt give xmax but gives width, so to get 
                # Xmax we add xmin and the width together
            
                # NOTE different image annotation software will give different methods of 
                # Bounding box dimensions. Use the defined annotation software (https://annotate.officialstatistics.org/)
                # to avoid reprogramming. 
                xmin_new = xx
                xmax_new = xx + ww
                ymin_new = yy
                ymax_new = yy + hh
            
                # BB values can never be larger than the image dimensions 
                if (xmax_new > self.image_res) or (xmin_new > self.image_res) or (ymax_new > self.image_res) or (ymax_new > self.image_res):
                    continue
    
                ##################END BAD DATA CHECK##########################

                fullpath = os.path.sep.join([self.imagepth, filename]).replace("\\", "/")
                self.dataframe["filename"].append(fullpath)

                image = cv2.imread(fullpath)
                (height, width) = image.shape[:2]

                self.dataframe["width"].append(width)
                self.dataframe["height"].append(height)

                self.dataframe["class"].append(type)

                self.dataframe["xmin"].append(xmin_new)
                self.dataframe["ymin"].append(ymin_new)
                self.dataframe["xmax"].append(xmax_new )
                self.dataframe["ymax"].append(ymax_new)
        else:
            print("Incorrect file type: Annoations file is not .txt or .csv")
            
        
            
            
            
            
    def controlled_split(self, test_annot_path, train_annot_path):
        '''
        Controlled split if data needs to be split in controlled manner
        '''
        
        if not self.controlled:
            print("ERROR: Non-controlled split selected but Controlled Split function entered")
        
        ############## First split loop ##########################
        
        # Annotation lines are shuffled to ensure all classes are treated equally
        shuffle_lines(train_annot_path)
        ext = os.path.splitext(train_annot_path)[-1].lower()

        if ext == ".csv":
            lines = open(train_annot_path, newline='')
            rows = csv.reader(lines,delimiter=',' )
        
            for row in rows:
                #parsing .txt file to obtain key values 
                try:
                    val = int(row[1])
                except ValueError:
                    continue
                  
    
                (filename, size, delim, rc ,ri,vals, type) = row
                vals = vals.split(":")
            
                xx = int(vals[2].strip("\"':,y")) * self.scalefactorX
                yy = int(vals[3].strip("\"':,width")) * self.scalefactorY
                ww = int(vals[4].strip("\"':,height")) * self.scalefactorX
                hh = int(vals[5].strip("\"':,}")) * self.scalefactorY
                type = type.strip("\"':{}")

                if (xx < 0) or (yy< 0) or (ww< 0) or (hh < 0):
                    print("error bounding box has negative value")
                    break
    
                ####################BAD DATA CHECK#############################
                '''
                tensorflow expects boundig boxes to be created from topleft to bottom right
                if data does not follow this error will be thrown
                np.nim and np.max solve this issue programically 
                '''
                # Data from labling doesnt give xmax but gives width, so to get 
                # Xmax we add xmin and the width together
            
                # NOTE different image annotation software will give different methods of 
                # Bounding box dimensions. Use the defined annotation software (https://annotate.officialstatistics.org/)
                # to avoid reprogramming. 
                xmin_new = xx
                xmax_new = xx + ww
                ymin_new = yy
                ymax_new = yy + hh
            
                if (xmax_new > self.image_res) or (xmin_new > self.image_res) or (ymax_new > self.image_res) or (ymax_new > self.image_res):
                    continue
    
                ##################END BAD DATA CHECK##########################
                fullpath = os.path.sep.join([self.imagepth, filename]).replace("\\", "/")
                self.dataframe["filename"].append(fullpath)

                image = cv2.imread(fullpath)
                (height, width) = image.shape[:2]

                self.dataframe["width"].append(width)
                self.dataframe["height"].append(height)

                self.dataframe["class"].append(type)

                self.dataframe["xmin"].append(xmin_new)
                self.dataframe["ymin"].append(ymin_new)
                self.dataframe["xmax"].append(xmax_new )
                self.dataframe["ymax"].append(ymax_new)
                
                
                
        elif ext == ".txt":
            rows = open(train_annot_path).read().strip().split("\n")
        
            for row in rows:
                #parsing .txt file to obtain key values 
                row=row.split(",")
                try:
                    tst = int(row[1])
                except ValueError:
                    continue
    
                (filename, size, delim, rc ,ri,shape, x, y, w, h, type) = row
                
                xx = int(x.strip("\"':x")) * self.scalefactorX
                yy = int(y.strip("\"':y")) * self.scalefactorY
                ww = int(w.strip("\"':width")) * self.scalefactorX
                hh = int(h.strip("\"':height}")) * self.scalefactorY
                type = type.strip("\"':{}")

                if (xx < 0) or (yy< 0) or (ww< 0) or (hh < 0):
                    print("error bounding box has negative value")
                    break
    
                ####################BAD DATA CHECK#############################
                '''
                tensorflow expects boundig boxes to be created from topleft to bottom right
                if data does not follow this error will be thrown
                np.nim and np.max solve this issue programically 
                '''
                # Data from labling doesnt give xmax but gives width, so to get 
                # Xmax we add xmin and the width together
            
                # NOTE different image annotation software will give different methods of 
                # Bounding box dimensions. Use the defined annotation software (https://annotate.officialstatistics.org/)
                # to avoid reprogramming. 
                xmin_new = xx
                xmax_new = xx + ww
                ymin_new = yy
                ymax_new = yy + hh
            
                # BB values can never be larger than the image dimensions 
                if (xmax_new > self.image_res) or (xmin_new > self.image_res) or (ymax_new > self.image_res) or (ymax_new > self.image_res):
                    continue
    
                ##################END BAD DATA CHECK##########################

                fullpath = os.path.sep.join([self.imagepth, filename]).replace("\\", "/")
                self.dataframe["filename"].append(fullpath)

                image = cv2.imread(fullpath)
                (height, width) = image.shape[:2]

                self.dataframe["width"].append(width)
                self.dataframe["height"].append(height)

                self.dataframe["class"].append(type)

                self.dataframe["xmin"].append(xmin_new)
                self.dataframe["ymin"].append(ymin_new)
                self.dataframe["xmax"].append(xmax_new )
                self.dataframe["ymax"].append(ymax_new)
        else:
            print("Incorrect file type: Annoations file is not .txt or .csv")
             
            
        ################## Second split loop ########################
        
        # Annotation lines are shuffled to ensure all classes are treated equally
        shuffle_lines(test_annot_path)
        ext = os.path.splitext(test_annot_path)[-1].lower()

        if ext == ".csv":
            lines = open(test_annot_path, newline='')
            rows = csv.reader(lines,delimiter=',' )
        
            for row in rows:
                #parsing .txt file to obtain key values 
                try:
                    val = int(row[1])
                except ValueError:
                    continue
                  
    
                (filename, size, delim, rc ,ri,vals, type) = row
                vals = vals.split(":")
            
                xx = int(vals[2].strip("\"':,y")) * self.scalefactorX
                yy = int(vals[3].strip("\"':,width")) * self.scalefactorY
                ww = int(vals[4].strip("\"':,height")) * self.scalefactorX
                hh = int(vals[5].strip("\"':,}")) * self.scalefactorY
                type = type.strip("\"':{}")

                if (xx < 0) or (yy< 0) or (ww< 0) or (hh < 0):
                    print("error bounding box has negative value")
                    break
    
                ####################BAD DATA CHECK#############################
                '''
                tensorflow expects boundig boxes to be created from topleft to bottom right
                if data does not follow this error will be thrown
                np.nim and np.max solve this issue programically 
                '''
                # Data from labling doesnt give xmax but gives width, so to get 
                # Xmax we add xmin and the width together
            
                # NOTE different image annotation software will give different methods of 
                # Bounding box dimensions. Use the defined annotation software (https://annotate.officialstatistics.org/)
                # to avoid reprogramming. 
                xmin_new = xx
                xmax_new = xx + ww
                ymin_new = yy
                ymax_new = yy + hh
            
                if (xmax_new > self.image_res) or (xmin_new > self.image_res) or (ymax_new > self.image_res) or (ymax_new > self.image_res):
                    continue
    
                ##################END BAD DATA CHECK##########################

                fullpath = os.path.sep.join([self.imagepth, filename]).replace("\\", "/")
                self.dataframe_test["filename"].append(fullpath)

                image = cv2.imread(fullpath)
                (height, width) = image.shape[:2]

                self.dataframe_test["width"].append(width)
                self.dataframe_test["height"].append(height)

                self.dataframe_test["class"].append(type)

                self.dataframe_test["xmin"].append(xmin_new)
                self.dataframe_test["ymin"].append(ymin_new)
                self.dataframe_test["xmax"].append(xmax_new )
                self.dataframe_test["ymax"].append(ymax_new)
                
                
                
        elif ext == ".txt":
            rows = open(test_annot_path).read().strip().split("\n")
        
            for row in rows:
                #parsing .txt file to obtain key values 
                row=row.split(",")
                
                # Ensure that line contains data
                try:
                    tst = int(row[1])
                except ValueError:
                    continue
                
                (filename, size, delim, rc ,ri,shape, x, y, w, h, type) = row
                
                xx = int(x.strip("\"':x")) * self.scalefactorX
                yy = int(y.strip("\"':y")) * self.scalefactorY
                ww = int(w.strip("\"':width")) * self.scalefactorX
                hh = int(h.strip("\"':height}")) * self.scalefactorY
                type = type.strip("\"':{}")

                if (xx < 0) or (yy< 0) or (ww< 0) or (hh < 0):
                    print("error bounding box has negative value")
                    break
    
                ####################BAD DATA CHECK#############################
                '''
                tensorflow expects boundig boxes to be created from topleft to bottom right
                if data does not follow this error will be thrown
                np.nim and np.max solve this issue programically 
                '''
                # Data from labling doesnt give xmax but gives width, so to get 
                # Xmax we add xmin and the width together
            
                # NOTE different image annotation software will give different methods of 
                # Bounding box dimensions. Use the defined annotation software (https://annotate.officialstatistics.org/)
                # to avoid reprogramming. 
                xmin_new = xx
                xmax_new = xx + ww
                ymin_new = yy
                ymax_new = yy + hh
            
                # BB values can never be larger than the image dimensions 
                if (xmax_new > self.image_res) or (xmin_new > self.image_res) or (ymax_new > self.image_res) or (ymax_new > self.image_res):
                    continue
    
                ##################END BAD DATA CHECK##########################

                fullpath = os.path.sep.join([self.imagepth, filename]).replace("\\", "/")
                self.dataframe_test["filename"].append(fullpath)

                image = cv2.imread(fullpath)
                (height, width) = image.shape[:2]

                self.dataframe_test["width"].append(width)
                self.dataframe_test["height"].append(height)

                self.dataframe_test["class"].append(type)

                self.dataframe_test["xmin"].append(xmin_new)
                self.dataframe_test["ymin"].append(ymin_new)
                self.dataframe_test["xmax"].append(xmax_new )
                self.dataframe_test["ymax"].append(ymax_new)
        else:
            print(f"Incorrect file type: {ext} \nAnnotations file is not .txt or .csv")
                
            
            
            
            
    def create_dataframe(self):
        '''
        Creates pandas dataframe for simple conversion to TFRECORD
        '''
        
        columb_name =  ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

        if self.controlled:
            self.df_test = pd.DataFrame(self.dataframe_test, columns=columb_name)
            self.df_train = pd.DataFrame(self.dataframe, columns=columb_name)
            
        else:
            self.df = pd.DataFrame(self.dataframe, columns=columb_name)
            
            
    def create_tf_records(self):
        '''
        Creates tf record using generate_tfrecord.py
        '''
        
        if self.controlled:
            generatecsv(self.df_train, self.train_record, self.imagepth, self.lablepth)
            generatecsv(self.df_test, self.test_record, self.imagepth, self.lablepth)
            
        else:
            train_df, test_df = train_test_split(self.df)
            generatecsv(train_df, self.train_record, self.imagepth, self.lablepth)
            generatecsv(test_df, self.test_record, self.imagepth, self.lablepth)
            
            
            
    def runner(self, steps, test_annot_path=None, train_annot_path=None):
        '''
        Runner function to being training. If controlled split is selected train and test annotations must be provided.
        Args:
            steps,            INT corrisponding to total training steps (Note: pipeline.config train.steps must match or be greater than this value)
            test_annot_path,  FILE PATH pointing to testing data annotations .txt file
            train_annot_path, FILE PATH pointing to testing data annotations .txt file
        '''
        
        if self.controlled:
            try:
                self.controlled_split(test_annot_path, train_annot_path)
            except Exception as e:
                logger.error(f"Data Splitting Failed: {e}")
        else:
            try:
                self.uncontrolled_split()
            except Exception as e:
                logger.error(f"Data Split Failed: {e}")


        try:
            self.create_dataframe()
        except Exception as e:
            logger.error(f"Error Creating Data Frame: {e}")
            self.Error_During_Training = True
            

        try:    
            self.create_tf_records()
        except Exception as e:
            logger.error(f"Error Converting to TF Record: {e}")
            self.Error_During_Training = True
            
        try:
            if self.selector == 1:
                try:
                    self.pipe_edit.open_configs(self.configpth, self.trainpth, self.testpth, self.lablepth, self.modelpth)
                except Exception as e:
                    logger.error(f"Error Occured Writing to FRCNN pipeline file: {e}")
                    print(f"Error Occured in FRCNN pipeline edit: {e}")
            elif self.selector == 2:
                try:
                    self.pipe_edit.open_configs_centernet(self.configpth, self.trainpth, self.testpth, self.lablepth, self.modelpth)
                except Exception as e:
                    logger.error(f"Error Occured Writing to Centernet pipeline file: {e}")
                    print(f"Error Occured in Centernet pipeline edit: {e}")
            elif self.selector == 3:
                try:
                    self.pipe_edit.open_configs_ssd(self.configpth, self.trainpth, self.testpth, self.lablepth, self.modelpth)
                except Exception as e:
                    logger.error(f"Error Occured Writing to SSD pipeline file: {e}")
                    print(f"Error Occured in SSD pipeline edit: {e}")
            elif self.selector == 4:
                try:
                    self.pipe_edit.open_configs_EffDet(self.configpth, self.trainpth, self.testpth, self.lablepth, self.modelpth)
                except Exception as e:
                    logger.error(f"Error Occured Writing to EFFDET pipeline file: {e}")
                    print(f"Error Occured in EFFDET pipeline edit: {e}")
            else:
                logger.error("Model selection incorrect value")
                print("Model Selector incorrect value. Process Terminated")
                return
        except Exception as e:
            logger.error(f"Error Editing Pipeline.config File: {e}")
        
        TRAINING_SCRIPT = os.path.join(self.apimodelpth, 'research', 'object_detection', 'model_main_tf2.py').replace("\\","/")

        self.configpth = self.configpth.replace("\\","/")
        
        self.exepath = self.apimodelpth.replace("tensorflow", "").replace("models-master", "").replace("//","/") + "model_main_tf2.exe"
        command = ["python", TRAINING_SCRIPT, "--model_dir=", self.modelpth, "--pipeline_config_path=", self.configpth, "--num_train_steps=", str(steps)]
        #NOTE for UNIX systems add preexec_fn=os.setsid to the subprocess arguments
        try:
            '''self.runproc = subprocess.Popen([self.exepath, 
                        f"--model_dir={self.modelpth}", 
                        f"--pipeline_config_path={self.configpth}",
                        f"--num_train_steps={str(int(float(steps)))}"], 
                        stdout=subprocess.PIPE, universal_newlines=True,  stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)'''
            self.runproc = subprocess.Popen(["python", TRAINING_SCRIPT, 
                        f"--model_dir={self.modelpth}", 
                        f"--pipeline_config_path={self.configpth}",
                        f"--num_train_steps={str(int(float(steps)))}"], stdout=subprocess.PIPE, universal_newlines=True,  stderr=subprocess.STDOUT)
        except Exception as e:
            logger.error(f"Train Running Failed: {e}")
            logger.info(f"path at failure: {self.exepath} \n Model Path: {self.modelpth}, \n Config Path: {self.configpth}")
            logger.info(f"Traceback: {traceback.print_exc()}")
            self.Error_During_Training = True
            
            


        self.q = Queue()
        t = threading.Thread(target=self.output_queue, args=(self.runproc.stdout, self.q))
        t.daemon = True
        t.start()
        logger.info("Training Started")
        

    def eval_runner(self):
        '''
        Runner fuction for tensorboard Eval (use tensorboard --logdir==[model eval directory])
        '''
        
        TRAINING_SCRIPT = os.path.join(self.apimodelpth, 'research', 'object_detection', 'model_main_tf2.py')
        
        commandEval = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, self.modelpth, self.configpth, self.modelpth)
        self.eval = subprocess.Popen(self.exepath, creationflags=subprocess.CREATE_NO_WINDOW)
        #os.system(commandEval)
        
    # Filling model pipeline.config parameters
    def fill_pipeline(self, num_classes, batch_size, learning_rate, max_num_boxes, obj_localization_weight, obj_classification_weight, num_steps):
        self.pipe_edit.set_vars(num_classes, batch_size, learning_rate, max_num_boxes, obj_localization_weight, obj_classification_weight, num_steps)
        self.pipeline_filled_flag = True
        
    def output(self, lines=None):
        try: 
            out = self.q.get_nowait()
        except Exception as e:
            pass
        else:
            return out
        
    
    def terminate(self):
        try:
            #os.kill(self.runproc.pid, signal.CTRL_BREAK_EVENT)
            self.runproc.terminate()
        except:
            print("Model did not terminate properly")
            logger.error("Model Termination was not sucessful")
        
    def output_queue(self, out, queue):
        for line in iter(out.readline, b''):
            queue.put(line)
        out.close()

    def terminateEval(self):
        try:
            self.eval.terminate()
        except:
            print("Evaluation did not terminate properly")
            logger.error("Evaulation Termination was not sucessful")

    def getErrorStatus(self):
        return self.Error_During_Training
        


            
        
        
        
        