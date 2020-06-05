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

class Model_FacialLandmarkDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
	    DONE
        '''
        self.plugin = None
        self.network = None
        self.device = device
        self.extensions = extensions
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
        
        with open(os.path.join(self.output_path, 'facial_landmarks.txt'), 'w') as f:
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
        
        with open(os.path.join(self.output_path, 'facial_landmarks.txt'), 'a') as f:
            #f.write("inference_time: ")
            f.write(str( t_1 - t_0)+'\n')

	    # extract the useful tensor
        outputs = result[self.output_name]

	    # get the eye-boxes
        eyes, eye_coords = self.preprocess_output(outputs, image)

        return eyes, eye_coords
	


    def check_model(self):
        # check for unsupported layers
        supported_layers = self.plugin.query_network(network = self.model, device_name = self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        
        if len(unsupported_layers) != 0:
            logging.info("Model_FacialLandmarkDetection - Unsupported layers found: {}".format(unsupported_layers))
            return False
        
        return True


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        input_model_width = self.input_shape[3]
        input_model_height = self.input_shape[2]
        
        p_image = cv2.resize(image, (input_model_width, input_model_height))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(1, *p_image.shape)
        
        return p_image



    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # get width and height of the original image
        original_width = image.shape[1]
        original_height = image.shape[0]
        
        # shape of outputs is (1, 10, 1, 1); reshape it
        outputs = outputs.reshape(-1)

        eye_boxes  = []
        eye_coords = []
        pix = 15

        # get the left eye
        x_l = int(outputs[0] * original_width)
        y_l = int(outputs[1] * original_height)

        # get the right eye
        x_r = int(outputs[2] * original_width)
        y_r = int(outputs[3] * original_height)

        # left eye box
        left_eye = image[(y_l - pix):(y_l + pix), (x_l - pix):(x_l + pix)] # to crop image[y_range, x_range]
        eye_boxes.append(left_eye)
        eye_coords.append([x_l, y_l])
        #cv2.rectangle(image, (x_l - pix, y_l - pix), (x_l + pix, y_l + pix), (0, 55, 255), 1)

        # right eye box
        right_eye = image[(y_r - pix):(y_r + pix), (x_r - pix):(x_r + pix)] # to crop image[y_range, x_range]
        eye_boxes.append(right_eye)
        eye_coords.append([x_r, y_r])
        #cv2.rectangle(image, (x_r - pix, y_r - pix), (x_r + pix, y_r + pix), (0, 55, 255), 1)

        return eye_boxes, eye_coords

	
