import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
from matplotlib import pyplot as plt

'''
This class is used to evaluate a custom tensorflow trained model. 

args:
    Selector, Set to one if detecting from a video camera otherwise set to detect from a single image
    Modelname, STR name of model file 
    checkpoint_path, FOLDER PATH to selected models checkpoint folder (rename this from defaut to checkpoint0)
    pipeline_path, FILE PATH to model pipeline.config
    Lablepath, FILE PATH to lablemap in .pbtxt format 
    Checkpoint, INT value corrisponding to model checkpoint to evaluate from
    Show, BOOLEAN flag, TRUE to display image with detections
    Save Location, PATH to where user wants to save image data
    Imagepath, FILE PATH to image NONE if using camera capture

'''



class Detector():
    def __init__(self, selector, modelname, checkpoint_path, pipeline_path, lablepath, checkpoint, show, save_location=None, image_path=None) -> None:
        self.selector = selector
        self.modelname = modelname
        self.checkpoint_path = checkpoint_path
        self.pipeline_path = pipeline_path
        self.lablepath = lablepath
        self.image_path = image_path
        self.checkpoint = str(checkpoint)
        self.show = show
        self.video = ''
        self.save_location = save_location
        self.terminate = False
        
        self.configs = config_util.get_configs_from_pipeline_file(self.pipeline_path)
        self.detection_model = model_builder.build(model_config=self.configs['model'], is_training=False)
        
        self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        self.ckpt.restore(os.path.join(checkpoint_path, 'ckpt-'+ self.checkpoint).replace("\\","/")).expect_partial()
        
        self.category_index = label_map_util.create_category_index_from_labelmap(lablepath)
        
        
      
    @tf.function
    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections
        
    def runner(self, thresh):
        if self.selector:
            cap = cv2.VideoCapture(0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
        while not self.terminate:
            if self.selector:
                ret, frame = cap.read()
                image_np = np.array(frame)
            else:
                imgs = cv2.imread(self.image_path)
                # Ensure that image being read is in RBG format not OpenCV default of BGR
                imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
                image_np = np.array(imgs)
    
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = self.detect_fn(input_tensor)
    
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        self.category_index,
                        use_normalized_coordinates=True,
                        min_score_thresh=thresh,
                        max_boxes_to_draw=500,
                        agnostic_mode=False)
            self.boxes = detections['detection_boxes']
            self.box_scores = detections['detection_scores']
            
            if self.save_location is not None:
                self.save_img(image_np_with_detections)

            self.setimage(image_np_with_detections)
            
            if self.show:
                cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (640, 640)))
                

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    if self.selector:
                        cap.release()
                    cv2.destroyAllWindows()
                    break
         
            
    def get_boxes(self):
        '''
        Returns ALL found bounding boxes and their scores 
        '''
        return self.boxes, self.box_scores
    
    
    
    def get_points(self, thresh):
        '''
        Returns center point for found boxes. 
        Args:
            Thresh: FLOAT between 0 and 1 
                Threshold value for bounding box score (score must be greater than this too be displayed)

        '''
        
        points = []
        self.trays = []
        index = 0
        for i in self.box_scores:
            if i < thresh:
                index = index + 1
                continue
            # Tensorflow gives detection boxes with data shape: YMIN, XMIN, YMAX, XMAX 
            y1  = int((self.boxes[index:(index+1):1,0]) * 640)
            x   = int((self.boxes[index:(index+1):1,1]) * 640)
            y   = int((self.boxes[index:(index+1):1,2]) * 640)
            x1  = int((self.boxes[index:(index+1):1,3]) * 640)
            
            # Dont include the full setters just the place points
            if (y - y1) > 100 or (x1 - x) > 100:
                self.trays.append((x,y,x1,y1))
                index = index + 1
                continue
    
            center = int(x1 - ((x1-x)/2)), int(y - ((y-y1)/2))
            points.append(center)
            index = index + 1
        return points, self.trays
    
    def save_img(self, img):
        cv2.imwrite(os.path.join(self.save_location, "Detection.jpg"), img)

    def setimage(self, image):
        self.video = image
    
    def getimage(self):
        if len(self.video) < 1:
            return [1]
        return self.video
    
    def end_eval(self):
        self.terminate = True

#testing
'''
modelname = 'faster_rcnn'
checkpoint_path = os.path.join(os.path.expanduser('~'), 'Documents', 'programming','object_detection', 'DeepLearn', 'trained_models', 'checkpoints', modelname)
pipeline_path = configpath = os.path.join(os.path.expanduser('~'), 'Documents', 'programming','object_detection', 'DeepLearn', 'trained_models', 'checkpoints', modelname, 'pipeline.config')
lablepath = os.path.join(os.path.expanduser('~'), 'Documents', 'programming','object_detection', 'DeepLearn', 'lable_map_bottle.pbtxt')


configs = config_util.get_configs_from_pipeline_file(pipeline_path)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(checkpoint_path, 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


category_index = label_map_util.create_category_index_from_labelmap(lablepath)
Timg_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'bottle.jpg')

#cap = cv2.VideoCapture(0)
#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#while cap.isOpened(): 
while True:
    #ret, frame = cap.read()
    imgs = cv2.imread(Timg_path)
    image_np = np.array(imgs)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        #cap.release()
        cv2.destroyAllWindows()
        break
        '''