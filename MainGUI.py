
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer, QSortFilterProxyModel, QRegularExpression, QEvent, QModelIndex, pyqtSlot, QSettings, Qt
from PyQt5.QtWidgets import QFileSystemModel, QAction, QWidget, QFileDialog, QDialog, QAbstractItemView, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from create_Project import createproject
from PyQt5.QtGui import QPixmap, QImage
from Model_Train import model_trainer
from matplotlib.figure import Figure
from PyQt5 import uic, QtWidgets
from detect import downloadmodel
import matplotlib.pyplot as plt
from TF_Detect import Detector
from functools import partial
from finder import finder
import numpy as np
import find_files
import threading
import logging
import time
import sys
import os

'''
All GUI functionallity handeled here. This file is used to start and handle all
user input to the GUI. 

Multiple dialog window classes can be found at the top of the file. 
Dialogs are used for small simple operations. NOTE: The create
project dialog is a much more involved popup thus is seperated into its own file
see create_Project.py for details. 

During training losses and other parameters are updated to the main page using 
Qtimer inturupts. This is to ensure the page does not freeze during training.
'''


#Setup Logging to objdet.log file
# create logger with 'spam_application'
logger = logging.getLogger('Main')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('logGUI.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

'''
This class is used to display the trendview graph for losses.
'''
class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        
'''
Testing class (unused at runtime)
'''       
class Results(QObject):
    def __init__(self, val):
        self.val = val
# Detection Training Thread Worker Class 
class worker(QObject):
    finished = pyqtSignal()
    data = pyqtSignal(object)
    
    
    def __init__(self, model, steps, train=None, test=None):
        super().__init__()
        self.train = train
        self.test = test
        self.model = model
        self.steps = steps
        self.count = 0
    def run(self):
        try:
            self.model.runner(self.steps, self.test, self.train)
        except Exception as e:
            logger.error(f"Error Running Model {e}")
  
'''
Testing (not used at runtime)
'''      
class QTextEditLogger(logging.Handler, QObject):
    class Emitter(QObject):
        log = pyqtSignal(str)
       
    def __init__(self, parent):
        super().__init__()
        QObject.__init__(self)
        self.widget = QtWidgets.QPlainTextEdit(parent)
        self.emitter = QTextEditLogger.Emitter()
        
        self.emitter.log.connect(self.widget.appendPlainText)   
        
    def emit(self, record):
        msg = self.format(record)
        self.emitter.log.emit(msg)
        #print(msg)
                
        

# Eval Worker class   
'''
Class used to notify when the model training has completed
'''      
class worker_eval(QObject):
    finished_eval = pyqtSignal()
    done = pyqtSignal()
    def __init__(self, model, thresh, dia):
        super().__init__()
        self.model = model
        self.thresh = thresh
        self.dialog = dia
    def run(self):
        self.model.runner(self.thresh)
        self.finished_eval.emit()
        self.done.emit()
    
        

class worker_downloader(QObject):
    def __init__(self, downloader, link, location):
        super().__init__()
        self.downloader = downloader
        self.link = link
        self.location = location
    def download(self):
        self.downloader.downloadmodel(self.link, self.location)
    

class aboutGUI(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.dialog = uic.loadUi('About.ui', self)

        self.dialog.close_btn.clicked.connect(lambda: self.dialog.close())


# Class used to display eval image and video
class displayimage(QDialog):
    def __init__(self, save_path, model, parent = None):
        super().__init__(parent)
        self.model = model
    
        self.dialog = uic.loadUi('eval_dialog.ui', self)
        
        self.dialog.close_btn.clicked.connect(self.reject)
        
        self.timer = QTimer(self, interval=100, timeout=self.updateui)
        self.timer.start()
        self.updateui()

        
    def setimage(self, image):
        self.image = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.eval_view.setPixmap(QPixmap.fromImage(self.image))
        print("Setting Image")
    def updateui(self):
        if len(self.model.getimage()) > 1 :
            self.dialog.setimage(self.model.getimage())
            print("updating ui")
            self.dialog.close_btn.clicked.connect(lambda: self.stop())
    
    def stop(self):
        self.timer.stop()
        self.model.end_eval()
  
        
# ======================================================== #
# Class used to allow drag and drop from treeview to paths #
# ======================================================== #       
class MouseObserver(QObject):
    def __init__(self, widget):
        super(MouseObserver, self).__init__(widget)
        self._widget = widget
        self.widget.installEventFilter(self)
        self.mouse = 0

    @property
    def widget(self):
        return self._widget

    def eventFilter(self, obj, event):
        if obj is self.widget:
            if event.type() == QEvent.MouseButtonPress:
                self.mouse = 1
            elif event.type() == QEvent.MouseButtonRelease:
                time.sleep(0.2)
                self.mouse = 0
        return super(MouseObserver, self).eventFilter(obj, event)   
    
    def get_mouse(self):
        #print(self.mouse)
        return self.mouse    


# =============================================================================== #
# This class is used to create, display and handle GUI events. Using PYQT5 this   #
# program creates multiple different GUI widgets. Widget actions are handled and  #
# passed to the "get and set" class above.                                        #
# =============================================================================== #   
class App(QtWidgets.QMainWindow):

    

    def __init__(self):
        super().__init__()
        
        # initalization of loss logging arrays
        self.running_total = []
        self.rpn_localization_rt = []
        self.rpn_obj_rt = []
        self.bc_localization_rt = []
        self.reg_loss_rt = []
        self.graph_type = 0
        
        self.model_name = ''
        self.label_path = ''
        self.checkpoint_path = ''

        self.persistance = QSettings('Object Detection GUI', 'SS Programs')
       
        #Logger Setup
        logger.info("GUI Started")
        
        self.title = 'PyQt5 button - pythonspot.com'
        self.ui = uic.loadUi('ObjDet.ui', self)
        
        self.tensor_path = os.path.join(os.getcwd(), 'tensorflow', 'models-master')
        
        # Top menu
        self.createMenubar()
        
        #Project tree view
        self.dragdata = None
        self.projectfolder = None
        self.mouse = 0 
        

        self.ui.folderAdd.clicked.connect
        self.projectPage()
        
        # PAGE 1 button click event handler 
        self.ui.page1_btn.clicked.connect(lambda: self.ui.formStackedWidget.setCurrentWidget(self.ui.modelselectpage))
        self.page1()

        # PAGE 2 button click event handler 
        self.ui.page2_btn.clicked.connect(lambda: self.ui.formStackedWidget.setCurrentWidget(self.ui.train_page))
        self.page2()

        # PAGE 3 button click event handler 
        self.ui.page3_btn.clicked.connect(lambda: self.ui.formStackedWidget.setCurrentWidget(self.ui.run_page))
        self.page3()
        
        # PAGE 4 (Pipeline config page) button click event handler 
        self.ui.page4_btn.clicked.connect(lambda: self.ui.formStackedWidget.setCurrentWidget(self.ui.Pipeline_page))
        self.page4()
        
        # PAGE 5 button click event handler 
        self.ui.eval_btn.clicked.connect(lambda: self.ui.formStackedWidget.setCurrentWidget(self.ui.Eval_page))
        self.page5()
        
        # Help page button click event handler
        self.ui.Help_Page_btn.clicked.connect(lambda: self.help())
        
        # Download page button click event handler
        # TODO: Depreciate this as create project is the prefered method
        self.ui.DownloadModelPageBtn.clicked.connect(lambda: self.ui.formStackedWidget.setCurrentWidget(self.ui.Download_Model))
        self.downloadpg()

        
        self.show()
        
        # Display 
        
        
    # ============================================== #
    # Function create/set/handle menu bar actions    #
    # ============================================== #
    def createMenubar(self):
        newAction = QAction("Open Folder", self)
        newAction.triggered.connect(self.openfolder)
        self.ui.menuFile.addAction(newAction)
        
        createAction = QAction("Create New Project", self)
        createAction.triggered.connect(self.createproj)
        self.ui.menuFile.addAction(createAction)

        propertiesAction = QAction("Edit Model Properties", self)
        propertiesAction.triggered.connect(lambda: self.ui.formStackedWidget.setCurrentWidget(self.ui.modeltrainpage))
        self.ui.menuProperties.addAction(propertiesAction)
        
        helpAction = QAction("Help", self)
        helpAction.triggered.connect(self.help)
        self.ui.menuHelp.addAction(helpAction)

        AboutAction = QAction("About", self)
        AboutAction.triggered.connect(self.About)
        self.ui.menuAbout.addAction(AboutAction)
    
    # ================================================= #
    # Function used to load a project fdlder into GUI   #
    # ================================================= #
    def openfolder(self):
        self.projectfolder = (QFileDialog.getExistingDirectory(self, 'Open File', os.path.expanduser("~")))
        logger.info(f"Opened Folder: {self.projectfolder}")
        self.addprojet()

    # ================================================= #
    # Function used to set the project directory        #
    # ================================================= #
    def setProjectDir(self):
        self.projectfolder = (QFileDialog.getExistingDirectory(self, 'Open File', os.path.expanduser("~")))
        logger.info(f"Set project directory to: {self.projectfolder}")
        self.persistance.setValue('project folder', self.projectfolder)

    # ================================================= #
    # Function used to call the create project dialog.  #
    # ================================================= #   
    def createproj(self):
        dialog = createproject(self)
        
        dialog.show()
        dialog.FinishedClicked.connect(self.autoimport)
    
    # ================================================= #
    # Function used to display about dialog.            #
    # ================================================= #  
    def About(self):
        dialog = aboutGUI(self)
        dialog.show()

    # ==================================================== #
    # With a project folder selected this function is used #
    # to find all file/folder paths needed to load/run the # 
    # model                                                #
    # ==================================================== #  
    def autoimport(self, folder):
        if folder == "Error":
            self.ui.Error_label.setText("Error creating project folder: See logs for details")
            return
        self.ui.Error_label.setText("Error: None")
        self.projectfolder = folder
        self.autofind()
        self.addprojet()

        # Updating project list
        self.projectPage()
        
    # ================================================================ #
    # Function used to load and display project folder in QTTreeView   #
    # ================================================================ #    
    def addprojet(self, folder = None):
        if folder is not None:
            path = folder
        else:
            path = self.projectfolder
        self.ui.ProjectView.setDragDropMode(QAbstractItemView.InternalMove)
        
        self.ui.ProjectView.clicked.connect(self.on_treeView_clicked)
        self.filelist = QFileSystemModel(self)
        self.filelist.setReadOnly(False)
        root_pth = self.filelist.setRootPath(path)
        self.filter = QSortFilterProxyModel(self.ui.ProjectView)
        
        self.filter.setSourceModel(self.filelist)
        
        self.filter.setFilterRegularExpression(QRegularExpression())
        self.ui.ProjectView.setModel(self.filter)
        self.ui.ProjectView.setColumnWidth(0, 300)
        self.ui.ProjectView.setRootIndex(self.filter.mapFromSource(root_pth))
        
        #folders = self.projectfolder.split("/")
        
        
    
    # ================================================================ #
    # Function used to allow users to drag from tree view into paths   #
    # ================================================================ #
    @pyqtSlot(QModelIndex)
    def on_treeView_clicked(self, index):
        index = self.filter.mapToSource(index)
        indexItem = self.filelist.index(index.row(), 0, index.parent())

        fileName = self.filelist.fileName(index)
        filePath = self.filelist.filePath(indexItem)

        self.dragdata = filePath
        
    # Function used to automate path finding for files/folders needed to 
    # load the model training program properly. If create project used this
    # is called automatically and all empty paths should be filled.
    def autofind(self, folder = None):
        if folder is not None:
            searchfolder = folder
        elif self.projectfolder is not None:
            searchfolder = self.projectfolder
        else:
            searchfolder = QFileDialog.getExistingDirectory(self, 'Open File', os.path.expanduser("~"))
        find = finder(searchfolder)
        locations = find.get_locations()
        
        self.pipeline = locations[2]
        #Setting model parameters 
        self.ui.image_path_3.setText(     locations[0])
        self.ui.annot_path_3.setText(     locations[1])
        self.ui.train_record_3.setText(   locations[3])
        self.ui.test_record_3.setText(    locations[4])
        self.ui.result_path_3.setText(    locations[5])
        self.ui.label_path_3.setText(     locations[6])
        self.ui.modelname_3.setText(      locations[7])
        self.ui.model_path_3.setText(     locations[8])
        self.ui.test_path_3.setText(      locations[9])
        self.ui.train_path_3.setText(     locations[9])
        self.ui.source_img_width.setText( locations[10])
        self.ui.source_img_height.setText(locations[11])

        #Eval parameters
        self.model_name = locations[7]
        self.label_path = locations[6]
        self.checkpoint_path = locations[8]


    # ================================================================ #
    # Function used to select and load project on click                #
    # ================================================================ #
    def setProject(self, name):
        project_folder = os.path.join(self.persistance.value('project folder'), name)
        self.autofind(folder = project_folder)
        self.addprojet(folder = project_folder)

    # ================================================================ #
    # Function used to find and add all projects in directory to main 
    # model selection page.                                            #
    # ================================================================ # 
    #TODO add page if more than x folders exist 
    def projectPage(self):
        self.foldercount = 0
        self.folders = {}
        if self.persistance.contains('project folder'):
            for dir in os.listdir(self.persistance.value('project folder')):
                self.folders[dir] = QPushButton(dir)
                self.folders[dir].setFixedSize(100,100)
                self.folders[dir].setStyleSheet("background-color:#6793b5;") 
                self.modelgrid.addWidget(self.folders[dir], 1 + self.foldercount // 4, self.foldercount % 4, 1,1, Qt.AlignCenter)
                self.foldercount+=1

        else:
            self.btn  = QPushButton("Set Project Directory")
            self.modelgrid.addWidget(self.btn, 1, 3, 2,1,)
            self.btn.clicked.connect(lambda: self.setProjectDir())
        for i in self.folders:
            self.folders[i].clicked.connect(partial(self.setProject, i))


    # PAGE 1 use to fill all file paths for model train program
    def page1(self):
        # BUTTON CLICK FUNCTIONS ON PAGE 1 
        # Used to find all non-optional paths
        self.ui.AutoFind.clicked.connect(lambda: self.autofind())
        
        # loading folder/file paths 
        self.ui.model_name_btn.clicked.connect(lambda:        self.ui.modelname_3.setText((QFileDialog.getOpenFileName(self, 'Open File', os.path.expanduser("~")))[0]) )
        
        # Event filters used for drag and drop functionality 
        self.ui.modelname_3.installEventFilter(self)
        
        self.ui.image_path_btn.clicked.connect(lambda:        self.ui.image_path_3.setText((QFileDialog.getExistingDirectory(self, 'Open File', os.path.expanduser("~")))) )
        self.ui.image_path_3.installEventFilter(self)
        
        self.ui.model_path_btn.clicked.connect(lambda:        self.ui.model_path_3.setText((QFileDialog.getExistingDirectory(self, 'Open File', os.path.expanduser("~")))) )
        self.ui.model_path_3.installEventFilter(self)
        
        self.ui.test_path_btn.clicked.connect(lambda:         self.ui.test_path_3.setText((QFileDialog.getExistingDirectory(self, 'Open File', os.path.expanduser("~")))) )
        self.ui.test_path_3.installEventFilter(self)
        
        self.ui.train_path_btn.clicked.connect(lambda:        self.ui.train_path_3.setText((QFileDialog.getExistingDirectory(self, 'Open File', os.path.expanduser("~")))))
        self.ui.train_path_3.installEventFilter(self)
        
        self.ui.lable_path_btn.clicked.connect(lambda:        self.ui.label_path_3.setText((QFileDialog.getOpenFileName(self, 'Open File', os.path.expanduser("~")))[0]) )
        self.ui.label_path_3.installEventFilter(self)
        
        self.ui.apimodel_path_3.setText(self.tensor_path)
        self.ui.api_model_path_btn.clicked.connect( lambda:   self.ui.apimodel_path_3.setText((QFileDialog.getExistingDirectory(self, 'Open File', os.path.expanduser("~")))) )
        self.ui.apimodel_path_3.installEventFilter(self)
        
        self.ui.results_path_btn.clicked.connect(lambda:      self.ui.result_path_3.setText((QFileDialog.getOpenFileName(self, 'Open File', os.path.expanduser("~")))[0]) )
        self.ui.result_path_3.installEventFilter(self)
        
        self.ui.annot_path_btn.clicked.connect(lambda:        self.ui.annot_path_3.setText((QFileDialog.getOpenFileName(self, 'Open File', os.path.expanduser("~")))[0]) )
        self.ui.annot_path_3.installEventFilter(self)
        
        self.ui.test_record_path.clicked.connect(lambda:      self.ui.test_record_3.setText((QFileDialog.getOpenFileName(self, 'Open File', os.path.expanduser("~")))[0]) )
        self.ui.test_record_3.installEventFilter(self)
        
        self.ui.train_record_path_btn.clicked.connect(lambda: self.ui.train_record_3.setText((QFileDialog.getOpenFileName(self, 'Open File', os.path.expanduser("~")))[0]) )
        self.ui.train_record_3.installEventFilter(self)
        
        self.ui.Model_next_page.clicked.connect(lambda: self.ui.formStackedWidget.setCurrentWidget(self.ui.Pipeline_page))
        
    # Function to handle button clicks on the training setup page    
    def page2(self):
        self.ui.train_annot_btn.clicked.connect(lambda: self.ui.train_annot.setText((QFileDialog.getOpenFileName(self, 'Open File', os.path.expanduser("~")))[0]) )
        self.ui.train_annot.installEventFilter(self)
        
        self.ui.Test_annot_btn.clicked.connect(lambda: self.ui.test_annot.setText((QFileDialog.getOpenFileName(self, 'Open File', os.path.expanduser("~")))[0]) )
        self.ui.test_annot.installEventFilter(self)
        
        self.ui.load_model_btn.clicked.connect(lambda: self.loadmodel())

    # Function to handel button clicks on the running page
    def page3(self):
        self.ui.run_model_train_btn.clicked.connect(lambda: self.runmodel())
        self.ui.Total_loss_btn.clicked.connect(lambda: self.setGraph(0))
        self.ui.rpn_loc_btn.clicked.connect(lambda: self.setGraph(1))
        self.ui.rpn_obj_btn.clicked.connect(lambda: self.setGraph(2))
        self.ui.box_loss_btn.clicked.connect(lambda: self.setGraph(3))
        self.ui.reg_loss_btn.clicked.connect(lambda: self.setGraph(4))
        self.ui.stop_btn.clicked.connect(lambda: self.close())

    def page4(self):
        self.ui.pipeline_next_page.clicked.connect(lambda: self.ui.formStackedWidget.setCurrentWidget(self.ui.train_page))
        
    #Funciton to handle button clicks on the evaluation page
    def page5(self):
        self.ui.imgpath_eval_btn.clicked.connect(lambda: self.ui.image_path_eval.setText((QFileDialog.getOpenFileName(self, 'Open File', os.path.expanduser("~")))[0]))
        self.ui.image_path_eval.installEventFilter(self)
        
        self.ui.Save_path_btn.clicked.connect(lambda: self.ui.save_path.setText((QFileDialog.getExistingDirectory(self, 'Open File', os.path.expanduser("~")))))
        self.ui.eval_button.clicked.connect(lambda: self.evalmodel())
        
    # Function to handel clicks on the download page
    def downloadpg(self):
        self.ui.download_mdl_btn.clicked.connect(lambda: self.ui.DownloadLocation.setText((QFileDialog.getExistingDirectory(self, 'Open File', os.path.expanduser("~")))) )
        self.ui.DownloadLocation.installEventFilter(self)
        
        self.ui.Download_Model_2.clicked.connect(lambda: self.downloadmodel())

    # function to call the help PDF viewer    
    def help(self):
        logger.info("Help Screen Opened")
        os.startfile("Object_Detection_Help_Guide.pdf")
        
        
    # ======================================================== #
    # Function used to load the model into the model trainer   #
    # ======================================================== #
    def loadmodel(self):
        
        # Ensure only a single model is selected
        A = self.ui.FasterRcnn.isChecked()
        B = self.ui.Centernet.isChecked()
        C = self.ui.SSD.isChecked()
        D = self.ui.EffDet.isChecked()
        
        if (A and B) or (A and C) or (A and D) or (B and C) or (B and D) or (D and C):
            self.ui.Error_label.setText("Error: Two Model Types Cannot Be Selected")
            self.ui.running_status.setText('Status: Error')
            return
        
        elif self.ui.FasterRcnn.isChecked():
            self.model_type = 1
            
        elif self.ui.Centernet.isChecked():
            self.model_type = 2
            
        elif self.ui.SSD.isChecked():
            self.model_type = 3
            
        elif self.ui.EffDet.isChecked():
            self.model_type = 4 
            
        else:
            self.ui.Error_label.setText("Error: No Model Type Selected")
            self.ui.running_status.setText('Status: Error')
            return
            
        
        
        
        # These if/else statements catch any missing inputs
        if len(self.ui.modelname_3.text()) == 0:
            self.ui.Error_label.setText("Error: Model name input not filled out. Missing path")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.image_path_3.text()) == 0:
            self.ui.Error_label.setText("Error: Image path input not filled out. Missing path")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.model_path_3.text()) == 0:
            self.ui.Error_label.setText("Error: Model path input not filled out. Missing path")
            self.ui.running_status.setText('Status: Error')
        
        elif len(self.ui.test_path_3.text()) == 0:
            self.ui.Error_label.setText("Error: Test path input not filled out. Missing path")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.train_path_3.text()) == 0:
            self.ui.Error_label.setText("Error: Train path input not filled out. Missing path")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.label_path_3.text()) == 0:
            self.ui.Error_label.setText("Error: Lable path input not filled out. Missing path")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.result_path_3.text()) == 0:
            self.ui.Error_label.setText("Error: Results path input not filled out. Missing path")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.annot_path_3.text()) == 0:
            self.ui.Error_label.setText("Error: Annotations path input not filled out. Missing path")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.test_record_3.text()) == 0:
            self.ui.Error_label.setText("Error: Test record input not filled out. Missing path")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.train_record_3.text()) == 0:
            self.ui.Error_label.setText("Error: Train record input not filled out. Missing path")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.apimodel_path_3.text()) == 0:
            self.ui.Error_label.setText("Error: API path input not filled out. Missing path")
            self.ui.running_status.setText('Status: Error')
        
        elif len(self.ui.number_classes_input.text()) == 0:
            self.ui.Error_label.setText("Error: Number of classes not defined")
            self.ui.running_status.setText('Status: Error')
        
        elif len(self.ui.batch_size_input.text()) == 0:
            self.ui.Error_label.setText("Error: Batch size not defined")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.classification_weight.text()) == 0:
            self.ui.Error_label.setText("Error: Classification weight not defined")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.localization_weight.text()) == 0:
            self.ui.Error_label.setText("Error: Localization weight not defined")
            self.ui.running_status.setText('Status: Error')
        
        elif len(self.ui.max_num_boxes_input.text()) == 0:
            self.ui.Error_label.setText("Error: Max number of boxes not defined")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.LearningRate.text()) == 0:
            self.ui.Error_label.setText("Error: Learning rate is not defined")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.steps.text()) == 0:
            self.ui.Error_label.setText("Error: Number of steps is not defined")
            self.ui.running_status.setText('Status: Error')
            
        elif len(self.ui.source_img_width.text()) == 0:
            self.ui.Error_label.setText("Error: Source Image Width Not Set")
            self.ui.running_status.setText('Status: Error')
            
        elif len(self.ui.source_img_height.text()) == 0:
            self.ui.Error_label.setText("Error: Source Image Height Not Set")
            self.ui.running_status.setText('Status: Error')
            
        elif len(self.ui.train_img_width.text()) == 0:
            self.ui.Error_label.setText("Error: Train Image Dimensions Not Set")
            self.ui.running_status.setText('Status: Error')


        # Model loaded if all paths are valid
        else:
            try:
                pipeline, checkpoint = find_files.get_paths(self.ui.model_path_3.text())
                self.model = model_trainer(self.ui.modelname_3.text(),
                                   self.ui.image_path_3.text(),
                                   self.ui.model_path_3.text(),
                                   pipeline,
                                   self.ui.test_path_3.text(),
                                   self.ui.train_path_3.text(),
                                   self.ui.label_path_3.text(),
                                   self.ui.apimodel_path_3.text(),
                                   checkpoint,
                                   self.ui.result_path_3.text(),
                                   self.ui.annot_path_3.text(),
                                   self.ui.test_record_3.text(),
                                   self.ui.train_record_3.text(),
                                   self.ui.Controlled_3.isChecked(),
                                   int(self.ui.source_img_width.text()),
                                   int(self.ui.source_img_height.text()),
                                   int(self.ui.train_img_width.text()),
                                   self.model_type
                                   )
            
                self.model.fill_pipeline(int(self.ui.number_classes_input.text()), 
                                     int(self.ui.batch_size_input.text()), 
                                     float(self.ui.LearningRate.text()),
                                     int(self.ui.max_num_boxes_input.text()),
                                     float(self.ui.localization_weight.text()),
                                     float(self.ui.classification_weight.text()),
                                     int(self.ui.steps.text())
                                     )
                logger.info(f"Model Loaded Sucessfully")
            except (RuntimeError, Exception, OSError) as e:
                print(e)
                self.ui.running_status.setText('Status: ERROR')
                self.ui.Error_label.setText(f"Error: {e}")
                logger.error(f"Model Loading Failed: {e}")
                pass

            self.ui.running_status.setText('Status: Loaded')
            self.ui.Error_label.setText("Error: None")
    
    
    # ================================================= #
    # Function used to run the loaded model             #
    # ================================================= #
    def runmodel(self):
        
        # Setting up a callback times to update/change the running page graphs
        self.count = 0
        self.getdata = QTimer(self)
        # Graphs are updated every 500 ms
        self.getdata.setInterval(500)
        self.getdata.timeout.connect(self.datacallback)
        self.getdata.start()
        
        if len(self.ui.LearningRate.text()) == 0:
            self.ui.Error_label.setText("Error: Learning Rate not set")
            self.ui.running_status.setText('Status: Error')

        elif len(self.ui.steps.text()) == 0:
            self.ui.Error_label.setText("Error: Number of steps not set")
            self.ui.running_status.setText('Status: Error')

        if self.ui.Controlled_3.isChecked() and len(self.ui.steps.text()) != 0 and len(self.ui.LearningRate.text()) != 0:
            if len(self.ui.train_annot.text()) == 0:
                self.ui.Error_label.setText("Error: Train Annotations not given for controlled data split")
                self.ui.running_status.setText('Status: Error')

            elif len(self.ui.test_annot.text()) == 0:
                self.ui.Error_label.setText("Error: Test Annotations not given for controlled data split")
                self.ui.running_status.setText('Status: Error')
                self.model.runner(int(self.ui.steps.text()), float(self.ui.LearningRate.text()), )
            else:
                self.clear_data()

                self.thread = QThread()
                self.worker = worker(self.model, int(self.ui.steps.text()), self.ui.train_annot.text(), self.ui.test_annot.text() )
                self.worker.moveToThread(self.thread)
                self.thread.started.connect(self.worker.run)
                self.worker.finished.connect(lambda: self.close())
             
                self.ui.running_status.setText('Status: Running')
                self.ui.Error_label.setText("Error: None")

                try:
                    self.thread.start()
                    logger.info("Model Training Started")
                except (RuntimeError, Exception, OSError) as e:
                    print(e)
                    self.ui.running_status.setText('Status: ERROR')
                    self.ui.Error_label.setText(f"Error: {e}")
                    logger.error(f"Model Could not Train: {e}")
                    pass
                

        elif (not self.ui.Controlled_3.isChecked()) and len(self.ui.steps.text()) != 0 and len(self.ui.LearningRate.text()) != 0:
            self.clear_data()
            # Impelmenting multithreading so GUI doesnt freeze while training
            
            self.thread = QThread()
            self.worker = worker(self.model, int(self.ui.steps.text()))
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(lambda: self.close())
             
            self.ui.running_status.setText('Status: Running')
            self.ui.Error_label.setText("Error: None")
           
            try:
                self.thread.start()
                logger.info("Model Training Started")
            except (RuntimeError, Exception, OSError) as e:
                print(e)
                self.ui.running_status.setText('Status: ERROR')
                self.ui.Error_label.setText(f"Error: {e}")
                logger.error(f"Model Could not Train: {e}")
                pass
            
    # =========================================================== #
    # Function used to capture data output from tf model training #
    # =========================================================== #       
    def datacallback(self):
        output = self.model.output()
        try:
            self.plot()
        except Exception as e:
            logger.error(f"Data Plotting Failed: {e}")

        if output is not None:
            print(f"Captured: {output}")
            self.set_Runtime_Info(output)
            self.ui.outputlogs.appendPlainText(output)
        
    # =========================================================== #
    # Function used to evaluate model on a image/camera stream    #
    # =========================================================== #  
    def evalmodel(self):
        if len(self.model_name) < 1:
            self.ui.Error_label.setText("Error: EVALUATION - Model name input not filled out. Missing path")
            self.ui.eval_status.setText('Status: Error')
            
        elif len(self.checkpoint_path) < 1:
            self.ui.Error_label.setText("Error: EVALUATION - Checkpoint path input not filled out. Missing path")
            self.ui.eval_status.setText('Status: Error')
            
        elif len(self.pipeline) < 1:
            self.ui.Error_label.setText("Error: EVALUATION - Pipeline path input not filled out. Missing path")
            self.ui.eval_status.setText('Status: Error')
        
        elif len(self.label_path) < 1:
            self.ui.Error_label.setText("Error: EVALUATION - Label path input not filled out. Missing path")
            self.ui.eval_status.setText('Status: Error')
            
        elif len(self.ui.image_path_eval.text()) == 0:
            self.ui.Error_label.setText("Error: EVALUATION - Image not given. Missing path")
            self.ui.eval_status.setText('Status: Error')
        
        elif len(self.ui.checkpoint_num.text()) == 0:
            self.ui.Error_label.setText("Error: EVALUATION - Checkpoint number not given")
            self.ui.eval_status.setText('Status: Error')
            
        elif len(self.ui.threshold_val.text()) == 0:
            self.ui.Error_label.setText("Error: EVALUATION - Threshold value not set")
            self.ui.eval_status.setText('Status: Error')
            
        elif len(self.ui.save_path.text()) == 0:
            self.ui.Error_label.setText("Error: EVALUATION - Save Location not set")
            self.ui.eval_status.setText('Status: Error')

        elif self.ui.video_eval.isChecked() and self.ui.image_eval.isChecked():
            self.ui.Error_label.setText("Error: EVALUATION - Multiple evaluation methods selected.")
            self.ui.eval_status.setText('Status: Error')
        
        elif self.ui.video_eval.isChecked() and not self.ui.image_eval.isChecked():
            try:
                det = Detector(1, 
                       self.model_name,
                       self.checkpoint_path,
                       self.pipeline,
                       self.label_path,
                       int(self.ui.checkpoint_num.text()),
                       False,
                       self.ui.save_path.text(),
                       self.ui.image_path_eval.text()
                       )
                logger.info(f"Evaulating: {self.model_name} on image: {self.ui.image_path_eval.text()}")

                # Create New Thread for evaluation
                self.dia = self.dialog_open(det)
                self.thread2 = QThread()
                self.eval_worker = worker_eval(det, float(self.ui.threshold_val.text()), self.dia)
                self.eval_worker.moveToThread(self.thread2)
                self.thread2.started.connect(self.eval_worker.run)
            
                # Handle one eval has complete
                self.eval_worker.finished_eval.connect(self.thread2.quit)
            
                self.ui.eval_status.setText('Status: Evaluating')

                # Begin Thread
                try:
                    self.thread2.start()
                    logger.info("Evaulation Running")
                except (RuntimeError, Exception, OSError) as e:
                    print(e)
                    self.ui.running_status.setText('Status: ERROR')
                    self.ui.Error_label.setText(f"Error: {e}")
                    logger.error(f"Evaulation Failed While Running: {e}")
                    pass
                
            except (RuntimeError, Exception, OSError) as e:
                print(e)
                self.ui.running_status.setText('Status: ERROR')
                self.ui.Error_label.setText(f"Error: {e}")
                logger.error(f"Evaulation Loading Failed: {e}")
                return
            
        elif not self.ui.video_eval.isChecked() and self.ui.image_eval.isChecked():
            try:
                det = Detector(0, 
                       self.model_name,
                       self.checkpoint_path,
                       self.pipeline,
                       self.label_path,
                       int(self.ui.checkpoint_num.text()),
                       False,
                       self.ui.save_path.text(),
                       self.ui.image_path_eval.text()
                       )
                logger.info(f"Evaulating: {self.model_name} on image: {self.ui.image_path_eval.text()}")

                # Create New Thread for evaluation
                self.dia = self.dialog_open(det)
                self.thread2 = QThread()
                self.eval_worker = worker_eval(det, float(self.ui.threshold_val.text()), self.dia)
                self.eval_worker.moveToThread(self.thread2)
                self.thread2.started.connect(self.eval_worker.run)
            
                # Handle one eval has complete
                self.eval_worker.finished_eval.connect(self.thread2.quit)
            
                self.ui.eval_status.setText('Status: Evaluating')

                # Begin Thread
                try:
                    self.thread2.start()
                    logger.info("Evaulation Running")
                except (RuntimeError, Exception, OSError) as e:
                    print(e)
                    self.ui.running_status.setText('Status: ERROR')
                    self.ui.Error_label.setText(f"Error: {e}")
                    logger.error(f"Evaulation Failed While Running: {e}")
                    pass
                
            except (RuntimeError, Exception, OSError) as e:
                print(e)
                self.ui.running_status.setText('Status: ERROR')
                self.ui.Error_label.setText(f"Error: {e}")
                logger.error(f"Evaulation Loading Failed: {e}")
                return
            
        else:
            self.ui.running_status.setText('Status: ERROR')
            self.ui.Error_label.setText(f"Error")
            logger.critical(f"Unknown state acheived on Evaluation step.\n",
                            f"Model selected: {self.model_name}\n",
                            f"Checkpoint {int(self.ui.checkpoint_num.text())}\n")


            
            
            
    def downloader(self, link, location):
        logger.info(f"Downloading Model: {link} to Folder: {location}")
        if (link is not None) and (location is not None):
            threading.Thread(target=downloadmodel, args=(link, location)).start()
        
        
            
    def dialog_open(self, detector):
        self.ui.eval_status.setText('Status: Evaluation Complete')
        dialog = displayimage(self.ui.save_path.text().replace("\\","/"), detector, parent=self)
        dialog.show()
        return dialog
        
    # Function used to terminate the TF program when the GUI is exited
    def close(self):
        try:
            self.model.terminate()
            
            # Quitting the Thread
            try:
                self.thread.quit()
            except: 
                try:
                    self.thread.exit()
                except:
                    logger.error("QThread did not terminate Sucessfully")
            
            # Quitting the Qtimer
            try:
                self.getdata.stop()
            except:
                logger.error("QTimer did not terminate Sucessfully")


            logger.info("Model Training Termination Complete")
        except (Exception, RuntimeError, OSError) as e:
            print("Error: Model did not exit properly")
            logger.error(f"Model did not terminate sucessfully {e}")
        self.ui.running_status.setText('Status: Stopped')
        

    # =========================================================== #
    # Function used to display model training output values       #
    # =========================================================== #  
    def set_Runtime_Info(self, info):
        # Moving tf output to GUI model type if statment used as tf output differs depending on 
        # Model type
        if  "Value in checkpoint could not be found in the restored object" in info:
            self.close()
            print("Error: Model did train properly. Issue with past checkpoints")
            logger.error("Error: Model did train properly. Issue with past checkpoints")

            self.ui.running_status.setText('Status: ERROR')
            self.ui.Error_label.setText("Error: Model did train properly. Issue with past checkpoints")

        if self.model_type == 1:
            if "Loss/BoxClassifierLoss/localization_loss" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.BC_loc_loss_value.setText(str(parse2[1]))
                self.bc_localization_rt.append(float(parse2[1]))
            
            if "Loss/RPNLoss/localization_loss" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.RPN_loc_loss_value.setText(str(parse2[1]))
                self.rpn_localization_rt.append(float(parse2[1]))
            
            if "Loss/RPNLoss/objectness_loss" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.obj_loss_value.setText(str(parse2[1]))
                self.rpn_obj_rt.append(float(parse2[1]))

            if "Loss/regularization_loss" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.reg_loss_value.setText(str(parse2[1]))
                self.reg_loss_rt.append(float(parse2[1]))
        
            if "Loss/total_loss" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.total_loss_value.setText(str(parse2[1]))
                self.running_total.append(float(parse2[1]))
                
            if "learning_rate" in info:
                parse1 = info.split(":")
                self.ui.learning_rate_value.setText(str(parse1[1]))   
                
            if "INFO:tensorflow:Step" in info:
                parse1 = info.split(' ')
                self.ui.step_value.setText(str(parse1[1]))
                self.ui.step_time_value.setText(str(parse1[4]))

                if int(parse1[1]) == int(self.ui.steps.text()):
                    self.close()
                    print("Training Complete")
                    logger.info("Training Complete")

                    self.ui.running_status.setText('Status: Complete')
                    self.ui.Error_label.setText("Error: None")
                
                #Running model evaluation every X steps as defined by user
                # default 10,000
                if (int(parse1[1]) % int(self.ui.eval_steps_num.text())) == 0:
                    self.close()
                    logger.info("Training Pause. Evaluation Starting...")
                    self.model.eval_runner()

                    #TODO emit and watch for end of evaluation then runmodel()
                    time.sleep(10)
                    self.runmodel()
                    logger.info("Evaluation Complete. Restarting Model Training")



        # Centernet model type        
        if self.model_type == 2:
            if "Loss/box/offset" in info:
                parse1 = info.split("{")
                parse2 = parse1[1].split(":")
                self.ui.Box_Class_Localize.setText("Box Offset")
                self.ui.box_loss_btn.setText("Box Offset")
                self.ui.BC_loc_loss_value.setText(str(parse2[1].replace(",","")))
                self.bc_localization_rt.append(float(parse2[1].replace(",","")))
            
            if "Loss/box/scale" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.RPN_Localize.setText("Box Scale")
                self.ui.rpn_loc_btn.setText("Box Scale")
                self.ui.RPN_loc_loss_value.setText(str(parse2[1]))
                self.rpn_localization_rt.append(float(parse2[1]))

            if "Loss/object_center" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.Regularization.setText("Object Center")
                self.ui.reg_loss_btn.setText("Object Center")
                self.ui.reg_loss_value.setText(str(parse2[1]))
                self.reg_loss_rt.append(float(parse2[1]))
        
            if "Loss/total_loss" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.total_loss_value.setText(str(parse2[1]))
                self.running_total.append(float(parse2[1]))
                
            if "learning_rate" in info:
                parse1 = info.split(":")
                self.ui.learning_rate_value.setText(str(parse1[1]))   
                
            if "INFO:tensorflow:Step" in info:
                parse1 = info.split(' ')
                self.ui.step_value.setText(str(parse1[1]))
                self.ui.step_time_value.setText(str(parse1[4]))

                if int(parse1[1]) == int(self.ui.steps.text()):
                    self.close()
                    print("Training Complete")
                    logger.info("Training Complete")

                    self.ui.running_status.setText('Status: Complete')
                    self.ui.Error_label.setText("Error: None")
                #Running model evaluation every X steps as defined by user
                # default 10,000
                if (int(parse1[1]) % int(self.ui.eval_steps_num.text())) == 0:
                    self.close()
                    logger.info("Training Pause. Evaluation Starting...")
                    self.model.eval_runner()

                    #TODO emit and watch for end of evaluation then runmodel()
                    time.sleep(10)
                    self.runmodel()
                    logger.info("Evaluation Complete. Restarting Model Training")

        # SSD model type        
        if self.model_type == 3:
            if "Loss/classification_loss" in info:
                parse1 = info.split("{")
                parse2 = parse1[1].split(":")
                self.ui.Box_Class_Localize.setText("Classification")
                self.ui.box_loss_btn.setText("Classification Loss")
                self.ui.BC_loc_loss_value.setText(str(parse2[1].replace(",","")))
                self.bc_localization_rt.append(float(parse2[1].replace(",","")))
            
            if "Loss/localization_loss" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.RPN_Localize.setText("Localization")
                self.ui.rpn_loc_btn.setText("Localization")
                self.ui.RPN_loc_loss_value.setText(str(parse2[1]))
                self.rpn_localization_rt.append(float(parse2[1]))

            if "Loss/regularization_loss" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.reg_loss_value.setText(str(parse2[1]))
                self.reg_loss_rt.append(float(parse2[1]))
        
            if "Loss/total_loss" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.total_loss_value.setText(str(parse2[1]))
                self.running_total.append(float(parse2[1]))
                #self.plot()
                
            if "learning_rate" in info:
                parse1 = info.split(":")
                self.ui.learning_rate_value.setText(str(parse1[1]))   
                
            if "INFO:tensorflow:Step" in info:
                parse1 = info.split(' ')
                self.ui.step_value.setText(str(parse1[1]))
                self.ui.step_time_value.setText(str(parse1[4]))

                if int(parse1[1]) == int(self.ui.steps.text()):
                    self.close()
                    print("Training Complete")
                    logger.info("Training Complete")

                    self.ui.running_status.setText('Status: Complete')
                    self.ui.Error_label.setText("Error: None")

                #Running model evaluation every X steps as defined by user
                # default 10,000
                if (int(parse1[1]) % int(self.ui.eval_steps_num.text())) == 0:
                    self.close()
                    logger.info("Training Pause. Evaluation Starting...")
                    self.model.eval_runner()

                    #TODO emit and watch for end of evaluation then runmodel()
                    time.sleep(10)
                    self.runmodel()
                    logger.info("Evaluation Complete. Restarting Model Training")

        # EffDet model type        
        if self.model_type == 4:
            if "Loss/classification_loss" in info:
                parse1 = info.split("{")
                parse2 = parse1[1].split(":")
                self.ui.Box_Class_Localize.setText("Classification")
                self.ui.box_loss_btn.setText("Classification Loss")
                self.ui.BC_loc_loss_value.setText(str(parse2[1].replace(",","")))
                self.bc_localization_rt.append(float(parse2[1].replace(",","")))
            
            if "Loss/localization_loss" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.RPN_Localize.setText("Localization")
                self.ui.rpn_loc_btn.setText("Localization")
                self.ui.RPN_loc_loss_value.setText(str(parse2[1]))
                self.rpn_localization_rt.append(float(parse2[1]))

            if "Loss/regularization_loss" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.reg_loss_value.setText(str(parse2[1]))
                self.reg_loss_rt.append(float(parse2[1]))
        
            if "Loss/total_loss" in info:
                parse1 = info.split(",")
                parse2 = parse1[0].split(":")
                self.ui.total_loss_value.setText(str(parse2[1]))
                self.running_total.append(float(parse2[1]))
                #self.plot()
                
            if "learning_rate" in info:
                parse1 = info.split(":")
                self.ui.learning_rate_value.setText(str(parse1[1]))   
                
            if "INFO:tensorflow:Step" in info:
                parse1 = info.split(' ')
                self.ui.step_value.setText(str(parse1[1]))
                self.ui.step_time_value.setText(str(parse1[4]))

                if int(parse1[1]) == int(self.ui.steps.text()):
                    self.close()
                    print("Training Complete")
                    logger.info("Training Complete")

                    self.ui.running_status.setText('Status: Complete')
                    self.ui.Error_label.setText("Error: None")

                #Running model evaluation every X steps as defined by user
                # default 10,000
                if (int(parse1[1]) % int(self.ui.eval_steps_num.text())) == 0:
                    self.close()
                    logger.info("Training Pause. Evaluation Starting...")
                    self.model.eval_runner()

                    #TODO emit and watch for end of evaluation then runmodel()
                    time.sleep(10)
                    self.runmodel()
                    logger.info("Evaluation Complete. Restarting Model Training")

    # ================================================================== #
    # Function used to clear past data from graphs                       #
    # ================================================================== #  
    def clear_data(self):
        self.running_total       = []
        self.reg_loss_rt         = []
        self.rpn_localization_rt = []
        self.bc_localization_rt  = []
        self.rpn_obj_rt          = []

    # ================================================================== #
    # Function used to display real time plots of model training outputs #
    # ================================================================== #   
    def plot(self):
        # Stops the program from creating a new graph each call
        for i in reversed(range(self.graph.count())): 
            self.graph.itemAt(i).widget().setParent(None)
        
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        
        if self.graph_type == 0:
            sc.axes.plot(self.running_total, color='red')
            sc.axes.set_title("Total Loss")
        elif self.graph_type == 1:
            sc.axes.plot(self.rpn_localization_rt, color='red')
            sc.axes.set_title("RPN Localization")
        elif self.graph_type == 2:
            sc.axes.plot(self.rpn_obj_rt, color='red')
            sc.axes.set_title("RPN Objectness")
        elif self.graph_type == 3:
            sc.axes.plot(self.bc_localization_rt, color='red')
            sc.axes.set_title("Box Localization")
        elif self.graph_type == 4:
            sc.axes.plot(self.reg_loss_rt, color='red')
            sc.axes.set_title("Regularization")
                
        self.graph.addWidget(sc)
        
    def setGraph(self, value):
        self.graph_type = value
        
    # ================================================================ #
    # Function used to enable drag and dropping from treeview to path. #
    # ================================================================ #  
    def eventFilter(self, source, event):
        #if source is self.edit
        #print(self.dragdata)
        if self.dragdata is not None:
            if event.type() == QEvent.Enter:
                source.setText(self.dragdata)
                #source.dropEvent(self)
                self.dragdata = None
                
                #source.setText(self.dragdata)
                return QWidget.eventFilter(self, source, event)
        return False


            
app = QtWidgets.QApplication(sys.argv)
window = App()


app.exec()
# Terminating tensorflow process when window closed
window.close()

