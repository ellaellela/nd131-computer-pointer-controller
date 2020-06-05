'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import logging
import numpy as np
import os
import time

from openvino.inference_engine import IECore
from openvino.inference_engine import IENetwork

class Model_FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        '''
        TODO: Use this to set your instance variables.
	    DONE
        '''
        self.plugin = None
        self.network = None
        self.device = device
        self.extensions = extensions
        self.threshold = threshold
        self.output_path = "../outputs/"

        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape




    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
	    DONE
        '''
        self.plugin = IECore()
        
        if self.check_model() != True:
            logging.info("Checking whether extensions are available to add to IECore...")
            
            if (self.extensions != None) and ("CPU" in self.device):
                self.plugin.add_extension(self.extensions, self.device)
                logging.info("Extension added.")
            else:
                logging.error("No extensions available. Exiting with error.")
                exit(1)
        
        t_0 = time.time()
        self.network = self.plugin.load_network(network = self.model, device_name = self.device)
        t_1 = time.time()
        
        with open(os.path.join(self.output_path, 'face_detection.txt'), 'w') as f:
            f.write("model_load_time: ")
            f.write(str( t_1 - t_0)+'\n')

        


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        DONE
        '''
        # preprocess and prepare input
        p_image = self.preprocess_input(image)
        input_dict = {self.input_name:p_image}
        
        # run inference
        t_0 = time.time()
        result = self.network.infer(input_dict)
        t_1 = time.time()
        
        with open(os.path.join(self.output_path, 'face_detection.txt'), 'a') as f:
            #f.write("inference_time: ")
            f.write(str( t_1 - t_0)+'\n')

	    # extract the useful tensor
        outputs = result[self.output_name]

	    # get the face-boxes
        box_list = self.preprocess_output(outputs)
        coords, image_post = self.draw_outputs(box_list, image.copy())

	    # crop the face-images
        cropped_faces = self.crop_faces(coords, image.copy())

        return coords, cropped_faces




    def draw_outputs(self, boxes, image):
        '''
	    Function converts coordinates into original image coordinates
	    and draws bounding boxes
	    DONE
        '''
        # get width and height of the original image
        original_width = image.shape[1]
        original_height = image.shape[0]

        box_coordinates = []
        
        # for every detected face draw a corresponding box
        for box in boxes:
            # get the box coordinates in the image coordinates
            x_ul = int(box[0] * original_width) # upper-left x
            y_ul = int(box[1] * original_height) # upper-left y
            x_br = int(box[2] * original_width) # bottom-right x
            y_br = int(box[3] * original_height) # bottom-right y
            
            box_coordinates.append(tuple([x_ul, y_ul, x_br, y_br]))
            image = cv2.rectangle(image, (x_ul, y_ul), (x_br, y_br), (0,0,255), 1)

        return box_coordinates, image




    def crop_faces(self, coords, image):
        '''
	    Crop the detected faces in the image
	    DONE
        '''
        cropped_list = []
        
        # for every detected face
        for box in coords:
	    # crop the face and add it to the list
            cropped_face = image[box[1]:box[3], box[0]:box[2]] # to crop image[y_range, x_range]
            cropped_list.append(cropped_face)

        return cropped_list





    def check_model(self):
        # check for unsupported layers
        supported_layers = self.plugin.query_network(network = self.model, device_name = self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        
        if len(unsupported_layers) != 0:
            logging.info("Model_FaceDetection - Unsupported layers found: {}".format(unsupported_layers))
            return False
        
        return True




    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        DONE
        '''
        input_model_width = self.input_shape[3]
        input_model_height = self.input_shape[2]
        
        p_image = cv2.resize(image, (input_model_width, input_model_height))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(1, *p_image.shape)
        
        return p_image





    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        DONE
        '''
        # outputs have the following shape: [1, 1, N, 7], where N is the number of detections
        box_list = []
        
        # if there is more than one face detected, sort by the probability
        sorted(outputs[0,0,:,:], key=lambda x : -x[2])
            
	    # each box contains the following data:
	    # [image_id, label, conf, x_min, y_min, x_max, y_max]
        for box in outputs[0,0,:,:]:            
            # the highest class and probability of detected box
            box_class = int(box[1])
            box_score = box[2]

            if box_score > self.threshold:
		    # get the coordinates of the rectangle
                x_ul = box[3] # upper-left x
                y_ul = box[4] # upper-left y
                x_br = box[5] # bottom-right x
                y_br = box[6] # bottom-right y
                    
                # append current face-box coordinates into return list
                box_list.append(tuple([x_ul, y_ul, x_br, y_br]))
        
        return box_list
