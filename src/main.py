import argparse
import cv2
import logging
import numpy as np
import os
import sys
import time

from argparse import ArgumentParser

from input_feeder 	    import InputFeeder
from mouse_controller 	import MouseController

from face_detection 		    import Model_FaceDetection
from head_pose_estimation 	    import Model_HeadPoseEstimation
from facial_landmarks_detection import Model_FacialLandmarkDetection
from gaze_estimation 		    import Model_GazeEstimation


def build_argparser():
    """
    Parse command line arguments.
    """
    parser = ArgumentParser()

    parser.add_argument("-fd_m", "--facedetecionmodel", required=True, type=str,
                        help="Path to face detection model's xml file with a trained model.")
    parser.add_argument("-hp_m", "--headposeestimationmodel", required=True, type=str,
                        help="Path to head pose estimation model's xml file with a trained model.")
    parser.add_argument("-fl_m", "--faciallandmarksdetectionmodel", required=True, type=str,
                        help="Path to facial landmarks detection model's xml file with a trained model.")
    parser.add_argument("-ge_m", "--gazeestimationnmodel", required=True, type=str,
                        help="Path to gaze estimation model's xml file with a trained model.")

    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image (IMG) or video (VID) or camera (CAM)")

    parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the kernels impl.")

    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")

    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering (0.5 by default)")
    parser.add_argument("-vis", "--visualise", type=str, default="",
                        help="Visualise outputs: head (HEAD), eyes (EYES), gaze (GAZE) (no visualisation by default)"
                             "It can slow down the application.")
    return parser


def calculate_mouse_vector(head_angles, gaze_vec):
    '''
    Function to calculate a new mouse vector
    '''
    roll_angle = head_angles[0][2]
    
    # normalise the gaze_vector
    gaze_vec = gaze_vec / np.linalg.norm(gaze_vec)
    
    # calculate the rotation angles
    cs = np.cos( roll_angle * np.pi / 180.0 )
    sn = np.sin( roll_angle * np.pi / 180.0 )
    
    # calculate the mouse coordinates
    # we need to rotate it to re-align it in the xy-plane
    x = cs * gaze_vec[0] + sn * gaze_vec[1]
    y = -sn * gaze_vec[0] + cs * gaze_vec[1]
    z = gaze_vec[2]
    
    return [x, y, z]


def visualise_vector(coords, image, vector):
    '''
    Draw a vector from the center of every eye image
    '''
    scale = 40
    
    left_base = ( coords[0][0], coords[0][1] )
    left_tip = ( left_base[0] + int(scale * vector[0]), left_base[1] - int(scale * vector[1]) )
    
    right_base = ( coords[1][0], coords[1][1] )
    right_tip = ( right_base[0] + int(scale * vector[0]), right_base[1] - int(scale * vector[1]) )
    
    image = cv2.line( image, left_base, left_tip, (0, 0, 255), 2 )
    image = cv2.line( image, right_base, right_tip, (0, 0, 255), 2 )
    cv2.imshow("cropped_face", image)
    
    return image


def main():
    args = build_argparser().parse_args()
    logging.basicConfig(filename = '../outputs/logging.log', level = logging.DEBUG)

    # get the model objects
    faceDetObj 		  = Model_FaceDetection(model_name = args.facedetecionmodel, device = args.device, extensions = args.cpu_extension, threshold = args.prob_threshold)
    headPoseObj 	  = Model_HeadPoseEstimation(model_name = args.headposeestimationmodel, device = args.device, extensions = args.cpu_extension)
    facialLandmarkObj = Model_FacialLandmarkDetection(model_name = args.faciallandmarksdetectionmodel, device = args.device, extensions = args.cpu_extension)
    gazeEstimationObj = Model_GazeEstimation(model_name = args.gazeestimationnmodel, device = args.device, extensions = args.cpu_extension)
    
    # load the models
    faceDetObj.load_model()
    headPoseObj.load_model()
    facialLandmarkObj.load_model()
    gazeEstimationObj.load_model()

    # check if we have video or cam stream
    stream = None
    if args.input.upper() == "CAM":
        stream = "cam"
    else:
        stream = "video"

    # get the InputFeeder and MouseController objects
    feedObj 		    = InputFeeder(input_type = stream, input_file = args.input)
    MouseControllerObj 	= MouseController(precision = 'high', speed = 'fast')

    # start processing the video or cam stream frames
    frame_count = 0
    feedObj.load_data()
    
    for flag, frame in feedObj.next_batch():
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        frame_count += 1
        
        coords, cropped_faces = faceDetObj.predict(frame)
        
        # check if we have detected a face in the frame
        if type(cropped_faces[0]) == int: # faces are sorted starting with the highest probability
            logging.info("FaceDetection did not detect any face - skipping the frame {}".format(frame_count))
            continue
        
        head_angles       = headPoseObj.predict(cropped_faces[0]) # faces are sorted starting with the highest probability
        eyes, eyes_coords = facialLandmarkObj.predict(cropped_faces[0]) # faces are sorted starting with the highest probability
        gaze_vec          = gazeEstimationObj.predict(eyes[0], eyes[1], head_angles)
        
        # get the mouse pointer coordinates
        mouse_vec = calculate_mouse_vector(head_angles, gaze_vec)
        
        # visualise the outputs
        if args.visualise.upper() == "FACE":
            cv2.imshow("detected face", cropped_faces[0])
        elif  args.visualise.upper() == "EYES":
            pix = 15
            eyes_image = cropped_faces[0].copy()
            
            # left eye
            x_l = eyes_coords[0][0]
            y_l = eyes_coords[0][1]
            eyes_image = cv2.rectangle(eyes_image, (x_l - pix, y_l - pix), (x_l + pix, y_l + pix), (0, 55, 255), 1)
            
            # right eye
            x_r = eyes_coords[1][0]
            y_r = eyes_coords[1][1]
            eyes_image = cv2.rectangle(eyes_image, (x_r - pix, y_r - pix), (x_r + pix, y_r + pix), (0, 55, 255), 1)
            
            cv2.imshow("detected eyes", eyes_image)
        elif args.visualise.upper() == "GAZE":
            fin_image = visualise_vector(eyes_coords, cropped_faces[0], mouse_vec)
        
        # move mouse pointer
        MouseControllerObj.move(mouse_vec[0], mouse_vec[1])
    
    cv2.destroyAllWindows()




if __name__=='__main__':
    main()
