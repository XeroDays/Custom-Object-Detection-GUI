import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import os
'''
Fuction to edit pipeline.config parameters. Its also possible to edit the pipeline.config file directly 
but values defined here will take precident over any direct changes in the pipeline.config file.

Args:
    Passed from Model_Train.py
    num_classes = Number of classes in data set
    batch_size = batch size of images per epoch while training 
    learning_rate = amout a single nuron can change per epoch 
    max_num_boxes = maximum number of classified objects in a single image
    Obj_classification_Weight/Localization_weight = setting models reward for correct localization/classification of objects
    Num_Steps = number of epoches the model will train for
'''
class pipeline_editor():
    def set_vars(self, num_classes, batch_size, learning_rate, max_num_boxes, obj_localization_weight, obj_classification_weight, num_steps):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_num_boxes = max_num_boxes
        self.obj_localization_weight = obj_localization_weight
        self.obj_classification_weight = obj_classification_weight
        self.num_steps = num_steps
    def open_configs(self, configpath, trainpath, testpath, lablepath, modelpath):
        '''
        args:
            config_path,     FILE PATH pointing to pipeline.config file
            test_path,       FOLDER PATH to location of test.record
            train_path,      FOLDER PATH to location of test.record
            lable_path,      FILE PATH pointing to lablemap in .pbtxt form
            model_path,      FOLDER PATH to downloaded model folder
        '''
        config = config_util.get_configs_from_pipeline_file(configpath)

        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(configpath, "r") as f:                                                                                                                                                                                                                     
            proto_str = f.read()                                                                                                                                                                                                                                          
            text_format.Merge(proto_str, pipeline_config)  


        #  pipeline_config.model.XXXXXX need to be changed depending on model type #
        pipeline_config.model.faster_rcnn.num_classes = self.num_classes
        pipeline_config.train_config.batch_size = self.batch_size
    
        pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = self.learning_rate
        pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = self.num_steps
        pipeline_config.train_config.num_steps = self.num_steps
    
        #========================================#
        # BB Guess boxes aspect and scale values #
        #========================================#
        '''
        pipeline_config.model.faster_rcnn.first_stage_anchor_generator.grid_anchor_generator.aspect_ratios = 0.1
        pipeline_config.model.faster_rcnn.first_stage_anchor_generator.grid_anchor_generator.aspect_ratios = 0.5
        pipeline_config.model.faster_rcnn.first_stage_anchor_generator.grid_anchor_generator.aspect_ratios = 1.0
        pipeline_config.model.faster_rcnn.first_stage_anchor_generator.grid_anchor_generator.scale = 0.1
        pipeline_config.model.faster_rcnn.first_stage_anchor_generator.grid_anchor_generator.scale = 0.25
        pipeline_config.model.faster_rcnn.first_stage_anchor_generator.grid_anchor_generator.scale = 0.5
        pipeline_config.model.faster_rcnn.first_stage_anchor_generator.grid_anchor_generator.scale = 1.0
        '''
    
    
    
        # ============================================================== #
        # These lines are used to ensure that enough bounding boxes are  #
        # made to find all objects in a image. If these values are too   #
        # low model accuracy will suffer.                                #
        # ============================================================== #
        pipeline_config.train_config.max_number_of_boxes = self.max_num_boxes
    
        pipeline_config.eval_config.max_num_boxes_to_visualize = self.max_num_boxes
        pipeline_config.eval_config.num_visualizations = self.max_num_boxes
        pipeline_config.eval_input_reader[0].max_number_of_boxes = self.max_num_boxes
    
        pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_detections_per_class = self.max_num_boxes
        pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_total_detections = self.max_num_boxes
        pipeline_config.model.faster_rcnn.first_stage_max_proposals = self.max_num_boxes
    
    
    
    
    
        ###########REDUCE RAM LOAD#####################
        #divide or multiply by 2
        #pipeline_config.train_config.batch_queue_capacity = 120
        #pipeline_config.train_config.num_batch_queue_threads = 60
        #pipeline_config.train_config.prefetch_queue_capacity = 80
        #pipeline_config.train_input_reader.queue_capacity = 2
        #pipeline_config.train_input_reader.min_after_dequeue = 1
        #pipeline_config.train_input_reader.num_readers = 1
        ###############################################
    
    
        # IF training from scratch comment out these two lines
        pipeline_config.train_config.fine_tune_checkpoint = os.path.join(modelpath, 'checkpoint0', 'ckpt-0')
        pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    
    
        pipeline_config.train_input_reader.label_map_path= lablepath
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(trainpath, 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = lablepath
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(testpath, 'test.record')]
    
        # See here for reasoning https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/issues/7
        pipeline_config.model.faster_rcnn.second_stage_classification_loss_weight = self.obj_classification_weight
        pipeline_config.model.faster_rcnn.first_stage_localization_loss_weight = self.obj_localization_weight
        config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
        with tf.io.gfile.GFile(configpath, "wb") as f:                                                                                                                                                                                                                     
            f.write(config_text)  
        
        
        
        
        
    # ====================================== #
    # Use this function for centernet models #
    # ====================================== #        
    def open_configs_centernet(self, configpath, trainpath, testpath, lablepath, modelpath):
        config = config_util.get_configs_from_pipeline_file(configpath)

        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(configpath, "r") as f:                                                                                                                                                                                                                     
            proto_str = f.read()                                                                                                                                                                                                                                          
            text_format.Merge(proto_str, pipeline_config)  


        # pipeline_config.model.XXXXXX need to be changed depending on model type #
        pipeline_config.model.center_net.num_classes = self.num_classes
        pipeline_config.train_config.batch_size = self.batch_size
    
        pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = self.learning_rate
        pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = self.num_steps
        pipeline_config.train_config.num_steps = self.num_steps
    
        # ============================================================== #
        # These lines are used to ensure that enough bounding boxes are  #
        # made to find all objects in a image. If these values are too   #
        # low model accuracy will suffer.                                #
        # ============================================================== #
        pipeline_config.train_config.max_number_of_boxes = self.max_num_boxes
        pipeline_config.eval_config.max_num_boxes_to_visualize = self.max_num_boxes
        pipeline_config.eval_config.num_visualizations = self.max_num_boxes
        pipeline_config.eval_input_reader[0].max_number_of_boxes =self.max_num_boxes
    
        ###########REDUCE RAM LOAD#####################
        #       divide by 2 to reduce further         #
        #pipeline_config.train_config.batch_queue_capacity = 120
        #pipeline_config.train_config.num_batch_queue_threads = 60
        #pipeline_config.train_config.prefetch_queue_capacity = 80
        #pipeline_config.train_input_reader.queue_capacity = 2
        #pipeline_config.train_input_reader.min_after_dequeue = 1
        #pipeline_config.train_input_reader.num_readers = 1
        ###############################################
    
        # ===================================================== #
        # IF training from scratch comment out these two lines. #
        # Also any mention of fine_tune_checkpoint in model     # 
        # pipeline.config needs to be removed/commented out     #
        #====================================================== #
        pipeline_config.train_config.fine_tune_checkpoint = os.path.join(modelpath, 'checkpoint0', 'ckpt-0')
        pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    
    
        pipeline_config.train_input_reader.label_map_path= lablepath
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(trainpath, 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = lablepath
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(testpath, 'test.record')]


        config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
        with tf.io.gfile.GFile(configpath, "wb") as f:                                                                                                                                                                                                                     
            f.write(config_text)  
        
        
        
    # ====================================== #
    # Use this function for SSD models #
    # ====================================== #        
    def open_configs_ssd(self, configpath, trainpath, testpath, lablepath, modelpath):
        config = config_util.get_configs_from_pipeline_file(configpath)

        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(configpath, "r") as f:                                                                                                                                                                                                                     
            proto_str = f.read()                                                                                                                                                                                                                                          
            text_format.Merge(proto_str, pipeline_config)  


        # pipeline_config.model.XXXXXX need to be changed depending on model type #
        pipeline_config.model.ssd.num_classes = self.num_classes
        pipeline_config.train_config.batch_size = self.batch_size
    
        pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = self.learning_rate
        pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = self.num_steps
        pipeline_config.train_config.num_steps = self.num_steps
    
        # ============================================================== #
        # These lines are used to ensure that enough bounding boxes are  #
        # made to find all objects in a image. If these values are too   #
        # low model accuracy will suffer.                                #
        # ============================================================== #
        pipeline_config.train_config.max_number_of_boxes = self.max_num_boxes
        pipeline_config.eval_config.max_num_boxes_to_visualize = self.max_num_boxes
        pipeline_config.eval_config.num_visualizations = self.max_num_boxes
        pipeline_config.eval_input_reader.max_number_of_boxes = self.max_num_boxes
    
        pipeline_config.model.ssd.post_processing.batch_non_max_suppression.max_total_detections = self.max_num_boxes
        pipeline_config.model.ssd.post_processing.batch_non_max_suppression.max_detections_per_class = self.max_num_boxes
        pipeline_config.model.ssd.post_processing.batch_non_max_suppression.max_classes_per_detection = self.num_classes
    
    
        ###########REDUCE RAM LOAD#####################
        #       divide by 2 to reduce further         #
        #pipeline_config.train_config.batch_queue_capacity = 120
        #pipeline_config.train_config.num_batch_queue_threads = 60
        #pipeline_config.train_config.prefetch_queue_capacity = 80
        #pipeline_config.train_input_reader.queue_capacity = 2
        #pipeline_config.train_input_reader.min_after_dequeue = 1
        #pipeline_config.train_input_reader.num_readers = 1
        ###############################################
    
    
    
        pipeline_config.train_config.fine_tune_checkpoint = os.path.join(modelpath, 'checkpoint0', 'ckpt-0')
        pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    
    
        pipeline_config.train_input_reader.label_map_path= lablepath
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(trainpath, 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = lablepath
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(testpath, 'test.record')]


        config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
        with tf.io.gfile.GFile(configpath, "wb") as f:                                                                                                                                                                                                                     
            f.write(config_text)  
        
    def open_configs_EffDet(self, configpath, trainpath, testpath, lablepath, modelpath):
        config = config_util.get_configs_from_pipeline_file(configpath)

        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(configpath, "r") as f:                                                                                                                                                                                                                     
            proto_str = f.read()                                                                                                                                                                                                                                          
            text_format.Merge(proto_str, pipeline_config)  


        # pipeline_config.model.XXXXXX need to be changed depending on model type #
        pipeline_config.model.ssd.num_classes = self.num_classes
        pipeline_config.train_config.batch_size = self.batch_size
    
        pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = self.learning_rate
        pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = self.num_steps
        pipeline_config.train_config.num_steps = self.num_steps
    
        pipeline_config.model.ssd.post_processing.batch_non_max_suppression.max_total_detections = self.max_num_boxes
        pipeline_config.model.ssd.post_processing.batch_non_max_suppression.max_detections_per_class = self.max_num_boxes
        pipeline_config.model.ssd.post_processing.batch_non_max_suppression.max_classes_per_detection = self.num_classes
    
    
        ###########REDUCE RAM LOAD#####################
        #       divide by 2 to reduce further         #
        #pipeline_config.train_config.batch_queue_capacity = 30
        #pipeline_config.train_config.num_batch_queue_threads = 15
        #pipeline_config.train_config.prefetch_queue_capacity = 20
        #pipeline_config.train_input_reader.queue_capacity = 2
        #pipeline_config.train_input_reader.min_after_dequeue = 1
        #pipeline_config.train_input_reader.num_readers = 1
        ###############################################
        

        pipeline_config.train_config.max_number_of_boxes = self.max_num_boxes
        pipeline_config.train_input_reader.max_number_of_boxes = self.max_num_boxes
        pipeline_config.eval_config.num_visualizations = self.max_num_boxes
        pipeline_config.eval_config.max_num_boxes_to_visualize = self.max_num_boxes
   
    
        pipeline_config.train_config.fine_tune_checkpoint = os.path.join(modelpath, 'checkpoint0', 'ckpt-0')
        pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    
    
        pipeline_config.train_input_reader.label_map_path= lablepath
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(trainpath, 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = lablepath
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(testpath, 'test.record')]


        config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
        with tf.io.gfile.GFile(configpath, "wb") as f:                                                                                                                                                                                                                     
            f.write(config_text)  
