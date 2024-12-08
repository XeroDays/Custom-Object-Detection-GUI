from PyQt5.QtWidgets import QDialog, QFileDialog
from create_lable_map import lable_map_creator
from PyQt5.QtCore import pyqtSignal
from PyQt5 import uic, QtWidgets
from detect import downloadmodel
from PIL import Image, ImageOps
import threading
import logging
import shutil
import time
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

'''
This class is used to handle the creation of new projects using
a Dialog called on the users selection of "Create Project" from the 
Main window toolbar. 

Once completed the class will emit a signal with the project
folder to the main window.

Args:
    passing from main window supply the "Self" param


'''

class createproject(QDialog):
    FinishedClicked = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)

        
        logger.info("Create Project Selected")

        #ui = os.path.join(os.path.expanduser('~'), 'Documents', 'programming','other', 'Gui','createproj.ui')
        self.dialog = uic.loadUi('createproj.ui', self)

        self.model_type = 99
        
        self.dialog.savelocbtn.clicked.connect(lambda: self.dialog.savelocation.setText((QFileDialog.getExistingDirectory(self, 'Open File', os.path.expanduser("~")))))
        self.savepath = self.dialog.savelocation.text()
    
        self.dialog.newProjNextPg.clicked.connect(lambda: self.setup())
        self.page2()
        
        self.dialog.nextPage2.clicked.connect(lambda: self.dialog.stackedWidget.setCurrentWidget(self.dialog.createProjPage3))
        self.page3()
        
        # Ensure that only a single model can be selected for download
        self.dialog.frcnnrescheck.stateChanged.connect(lambda: self.single()) 
        self.dialog.effDetCheck.stateChanged.connect(  lambda: self.single()) 
        self.dialog.centercheck.stateChanged.connect(  lambda: self.single())
        self.dialog.ssdcheck.stateChanged.connect(     lambda: self.single())
        self.dialog.frcnncheck.stateChanged.connect(   lambda: self.single())
        
        # Ensure that only a single model size can be selected
        self.dialog.smallimagecheck.stateChanged.connect(lambda: self.single2())
        self.dialog.largeimagecheck.stateChanged.connect(lambda: self.single2())
        self.dialog.xLarge_image.stateChanged.connect(lambda: self.single2() )
        
        # models links are as follows: 
        # FRCNN INCEPTION RESNET 640, FRCNN INCEPTION RESNET 1024, FRCNN RESNET V2 640, FRCNN RESNET V2 1024, SSD 640, SSD 1024, CENTERNET 512, CENTERNET 1024
        self.model_links = ['http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz',
                            'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz',
                            'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz',
                            'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz',
                            'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz',
                            'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
                            'http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz',
                            'http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz',
                            'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz',
                            'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz',
                            'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz']

    # ============================================= #
    # Function for the first page of create project #
    # ============================================= #
    def setup(self):
        
        self.savepath = self.dialog.savelocation.text()
        self.fullpath = os.path.join(self.savepath, self.dialog.projectname.text()).replace("\\","/")
        
        # Allowing user to progress through the create project page without
        # defining a project name/save location
        try:
            os.mkdir(self.fullpath)
        except Exception as e:
            self.dialog.stackedWidget.setCurrentWidget(self.dialog.createProjPage2)
            return
        files = [os.path.join(self.fullpath, 'test.record'), os.path.join(self.fullpath, 'train.record'), os.path.join(self.fullpath, 'results.h5')]
            
        for i in files:
            with open(i, 'w'): pass
        
        self.annotspath = os.path.join(self.fullpath, 'Annotations')
        try:
            os.mkdir(self.annotspath)
        except Exception as e:
            self.dialog.stackedWidget.setCurrentWidget(self.dialog.createProjPage2)
            return
        
        self.imagespath = os.path.join(self.fullpath, 'Images')
        self.origin_Image_Path = os.path.join(self.imagespath, "Original")
        self.resized_Image_Path = os.path.join(self.imagespath, "Resized")
        try:
            os.mkdir(self.imagespath)
            os.mkdir(self.origin_Image_Path)
            os.mkdir(self.resized_Image_Path)
        except Exception as e:
            self.dialog.stackedWidget.setCurrentWidget(self.dialog.createProjPage2)
            return
        
        self.dialog.stackedWidget.setCurrentWidget(self.dialog.createProjPage2)
       
    # ================================================== #
    # Function to download model without freezing dialog #
    # ================================================== #
    def create_n_exit(self):
        self.dialog.stackedWidget.setCurrentWidget(self.dialog.createProjPage4)
        self.dialog.stackedWidget.repaint()
        # Three if statements below are to put the annotations into the correct folder
        if len(self.dialog.annots.text()) != 0:
            shutil.copy2(self.dialog.annots.text(), os.path.join(self.annotspath))
        
        if len(self.dialog.testAnnots.text()) != 0:
            shutil.copy2(self.dialog.testAnnots.text(), os.path.join(self.annotspath))
        
        if len(self.dialog.trainAnnots.text()) != 0:
            shutil.copy2(self.dialog.trainAnnots.text(), os.path.join(self.annotspath))
            
        # Model downloaded here
        if self.dialog.downloadcheck.isChecked():        
            modelpath = os.path.join(self.fullpath, (self.dialog.projectname.text()+'_Model'))
            
            try:
                os.mkdir(modelpath)
            except Exception as e:
                logger.error(f"Cannot Create Folder: Path Exists {e}")
                self.dialog.FinishedClicked.emit("Error")
                self.dialog.close()
                return

            checkpath = os.path.join(modelpath, 'trained_models', 'checkpoints', 'Detection_Model')

            # FRCNN RESnET CHECKED    
            if self.dialog.frcnnrescheck.isChecked() and self.dialog.smallimagecheck.isChecked():
                self.downloadmodel(self.model_links[0], modelpath)

                # While loop used to ensure user doesnt progress until model has
                # Downloaded Completely
                while not os.path.exists(checkpath):
                    time.sleep(0.5)

                #Autoloading new project   
                self.dialog.FinishedClicked.emit(self.fullpath)
                self.dialog.close()
                return
            elif self.dialog.frcnnrescheck.isChecked() and self.dialog.largeimagecheck.isChecked():
                self.downloadmodel(self.model_links[1], modelpath)

                # While loop used to ensure user doesnt progress until model has
                # Downloaded Completely
                while not os.path.exists(checkpath):
                    time.sleep(0.5)

                #Autoloading new project   
                self.dialog.FinishedClicked.emit(self.fullpath)
                self.dialog.close()
                return
            
            # FRCNN CHECKED 
            if self.dialog.frcnncheck.isChecked() and self.dialog.smallimagecheck.isChecked():
                self.downloadmodel(self.model_links[2], modelpath)

                # While loop used to ensure user doesnt progress until model has
                # Downloaded Completely
                while not os.path.exists(checkpath):
                    time.sleep(0.5)

                #Autoloading new project   
                self.dialog.FinishedClicked.emit(self.fullpath)
                self.dialog.close()
                return
            elif self.dialog.frcnncheck.isChecked() and self.dialog.largeimagecheck.isChecked():
                self.downloadmodel(self.model_links[3], modelpath)

                # While loop used to ensure user doesnt progress until model has
                # Downloaded Completely
                while not os.path.exists(checkpath):
                    time.sleep(0.5)

                #Autoloading new project   
                self.dialog.FinishedClicked.emit(self.fullpath)
                self.dialog.close()
                return
               
            # SSD CHECKED 
            if self.dialog.ssdcheck.isChecked() and self.dialog.smallimagecheck.isChecked():
                self.downloadmodel(self.model_links[4], modelpath)

                # While loop used to ensure user doesnt progress until model has
                # Downloaded Completely
                while not os.path.exists(checkpath):
                    time.sleep(0.5)

                #Autoloading new project   
                self.dialog.FinishedClicked.emit(self.fullpath)
                self.dialog.close()
                return
            elif self.dialog.ssdcheck.isChecked() and self.dialog.largeimagecheck.isChecked():
                self.downloadmodel(self.model_links[5], modelpath)

                # While loop used to ensure user doesnt progress until model has
                # Downloaded Completely
                while not os.path.exists(checkpath):
                    time.sleep(0.5)

                #Autoloading new project   
                self.dialog.FinishedClicked.emit(self.fullpath)
                self.dialog.close()
                return
             
            # CENTERNET CHECKED 
            if self.dialog.centercheck.isChecked() and self.dialog.smallimagecheck.isChecked():
                self.downloadmodel(self.model_links[6], modelpath)

                # While loop used to ensure user doesnt progress until model has
                # Downloaded Completely
                while not os.path.exists(checkpath):
                    time.sleep(0.5)

                #Autoloading new project   
                self.dialog.FinishedClicked.emit(self.fullpath)
                self.dialog.close()
                return
            elif self.dialog.centercheck.isChecked() and self.dialog.largeimagecheck.isChecked():
                self.downloadmodel(self.model_links[7], modelpath)

                # While loop used to ensure user doesnt progress until model has
                # Downloaded Completely
                while not os.path.exists(checkpath):
                    time.sleep(0.5)

                #Autoloading new project   
                self.dialog.FinishedClicked.emit(self.fullpath)
                self.dialog.close()
                return
                
            # EFFDET CHECKED 
            if self.dialog.effDetCheck.isChecked() and self.dialog.smallimagecheck.isChecked():
                self.downloadmodel(self.model_links[10], modelpath)

                # While loop used to ensure user doesnt progress until model has
                # Downloaded Completely
                while not os.path.exists(checkpath):
                    time.sleep(0.5)

                #Autoloading new project   
                self.dialog.FinishedClicked.emit(self.fullpath)
                self.dialog.close()
                return
            elif self.dialog.effDetCheck.isChecked() and self.dialog.largeimagecheck.isChecked():
                self.downloadmodel(self.model_links[9], modelpath)
                
                # While loop used to ensure user doesnt progress until model has
                # Downloaded Completely
                while not os.path.exists(checkpath):
                    time.sleep(0.5)

                #Autoloading new project   
                self.dialog.FinishedClicked.emit(self.fullpath)
                self.dialog.close()
                return
            elif self.dialog.effDetCheck.isChecked() and self.dialog.xLarge_image.isChecked():
                self.downloadmodel(self.model_links[8], modelpath)   
                
                # While loop used to ensure user doesnt progress until model has
                # Downloaded Completely
                while not os.path.exists(checkpath):
                    time.sleep(0.5)
                
                #Autoloading new project   
                self.dialog.FinishedClicked.emit(self.fullpath)
                self.dialog.close()
                return
            
            #self.dialog.close()
        else:
            self.dialog.close()
        
    # ========================================== #
    # Model downloader function                  #
    # ========================================== #
    def downloadmodel(self, link, location):
        logger.info("Model selected for download")

        logger.info(f"Downloading Model: {link} to Folder: {location}")
        if (link is not None) and (location is not None):
            threading.Thread(target=downloadmodel, args=(link, location)).start()
        # Emitter used to download the model from the main page 
        #self.dialog.FinishedClicked.emit(link, location)
        
        
        
    # ========================================== #
    # Function to handle user inputs on page 2   #
    # ========================================== #
    def page2(self):
        self.dialog.addImages.clicked.connect(lambda: self.image_list((QFileDialog.getOpenFileNames(self, 'Open File', os.path.expanduser("~")))[0]))
        self.dialog.addAnnots.clicked.connect(lambda: self.dialog.annots.setText((QFileDialog.getOpenFileName(self, 'Open File', os.path.expanduser("~")))[0]) )
        self.dialog.addTestAnnots.clicked.connect(lambda: self.dialog.testAnnots.setText((QFileDialog.getOpenFileName(self, 'Open File', os.path.expanduser("~")))[0]) )
        self.dialog.addTrainAnnots.clicked.connect(lambda: self.dialog.trainAnnots.setText((QFileDialog.getOpenFileName(self, 'Open File', os.path.expanduser("~")))[0]) )
        self.dialog.resize_btn.clicked.connect(lambda: self.imageresizer())

    # ========================================== #
    # Function to resize images to specific w/h  #
    # ========================================== #
    def imageresizer(self):
        
        imagepath = os.path.join(self.fullpath, 'Images', 'Resized')
        
        logger.info("Images Resized")
        if len(self.dialog.imageReSize.text()) > 0:
            for file in os.listdir(imagepath):

                if (file.endswith('.JPG') or file.endswith('.jpg') or 
                    file.endswith('.PNG') or file.endswith('.png') or 
                    file.endswith('.BMP') or file.endswith('.bmp')):

                    fullpath = os.path.join(imagepath, file)
    
                    img = Image.open(fullpath)

                    #Saving original image
                    origin_path  = os.path.join(self.origin_Image_Path, file)
                    img.save(origin_path)

                    img = ImageOps.exif_transpose(img)
                    img = img.resize((int(self.dialog.imageReSize.text()), int(self.dialog.imageReSize.text())))
                    logger.info(f"Image Resizing to {self.dialog.imageReSize.text()}")
    
                    img.save(fullpath)
                    
                else:
                    continue
    

    # ========================================== #
    # Function to handle user inputs on page 3   #
    # ========================================== #
    def page3(self):
        self.dialog.confirmLablesBtn.clicked.connect(lambda: self.update_lables())
            
        
        self.dialog.completeProjCreate.clicked.connect(lambda: self.create_n_exit())
    
    # ============================================== #
    # Function add/update/display lables on page 3   #
    # ============================================== #
    def update_lables(self):
        
        lables = self.dialog.lables.text().split(",")
        row = 0 
        column = 0
        self.dialog.lablesTable.setRowCount(len(lables))
        for i in lables:
            self.dialog.lablesTable.setItem(row, column, QtWidgets.QTableWidgetItem(i))
            row = row + 1
        try:
            lable_map_creator.create_lable_map(lables, self.fullpath)
        except Exception as e:
            print("Lable Map Already Exists in Directory")
    
    # ============================================== #
    # Function add/update/display images on page 2   #
    # ============================================== #
    def image_list(self, images):
        row = 0 
        column = 0
        self.dialog.imagesTable.setRowCount(len(images))
        for i in images:
            name = i.split('/')
            final = len(name) - 1 
            shutil.copy2(i, os.path.join(self.fullpath, 'Images', 'Resized'))
            self.dialog.imagesTable.setItem(row, column, QtWidgets.QTableWidgetItem(name[final]))
            row = row + 1

    
    # ======================================================== #
    # Function to ensure only a single model can be selected   #
    # ======================================================== #
    def single(self):
        if self.dialog.frcnnrescheck.isChecked():
            self.dialog.effDetCheck.setCheckable(False)
            self.dialog.centercheck.setCheckable(False)
            self.dialog.ssdcheck.setCheckable(False)
            self.dialog.frcnncheck.setCheckable(False)
            return
            
        elif self.dialog.effDetCheck.isChecked():
            self.dialog.frcnnrescheck.setCheckable(False)
            self.dialog.centercheck.setCheckable(False)
            self.dialog.ssdcheck.setCheckable(False)
            self.dialog.frcnncheck.setCheckable(False)
            return
            
        elif self.dialog.centercheck.isChecked():
            self.dialog.frcnnrescheck.setCheckable(False)
            self.dialog.effDetCheck.setCheckable(False)
            self.dialog.ssdcheck.setCheckable(False)
            self.dialog.frcnncheck.setCheckable(False)
            return
            
        elif self.dialog.ssdcheck.isChecked():
            self.dialog.frcnnrescheck.setCheckable(False)
            self.dialog.effDetCheck.setCheckable(False)
            self.dialog.centercheck.setCheckable(False)
            self.dialog.frcnncheck.setCheckable(False)
            return
            
        elif self.dialog.frcnncheck.isChecked():
            self.dialog.frcnnrescheck.setCheckable(False)
            self.dialog.effDetCheck.setCheckable(False)
            self.dialog.centercheck.setCheckable(False)
            self.dialog.ssdcheck.setCheckable(False)
            return
            
        # If all are unchecked
        self.dialog.effDetCheck.setCheckable(True)
        self.dialog.centercheck.setCheckable(True)
        self.dialog.ssdcheck.setCheckable(True)
        self.dialog.frcnncheck.setCheckable(True)
        self.dialog.frcnnrescheck.setCheckable(True)
        return
            
    # ================================================================= #
    # Function to ensure only a single model image size can be selected #
    # ================================================================= #    
    def single2(self):
        if self.dialog.smallimagecheck.isChecked():
            self.dialog.largeimagecheck.setCheckable(False)
            self.dialog.xLarge_image.setCheckable(False)
            return
            
        elif self.dialog.largeimagecheck.isChecked():
            self.dialog.smallimagecheck.setCheckable(False)
            self.dialog.xLarge_image.setCheckable(False)
            return
            
        elif self.dialog.xLarge_image.isChecked():
            self.dialog.smallimagecheck.setCheckable(False)
            self.dialog.largeimagecheck.setCheckable(False)
            return
            
        
        self.dialog.smallimagecheck.setCheckable(True)
        self.dialog.largeimagecheck.setCheckable(True)
        self.dialog.xLarge_image.setCheckable(True)
        return