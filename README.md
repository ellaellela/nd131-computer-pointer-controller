# Computer Pointer Controller
This is a project where we have built an application using openVINO to control the mouse pointer with the gaze. In total, four pretrained models have been used to acheive this:

1) face detection model (face-detection-adas-binary-0001)
2) head-pose estimation model (head-pose-estimation-adas-0001)
3) facial landmarks detection model (landmarks-regression-retail-0009)
4) gaze estimation model (gaze-estimation-adas-0002)

Input: Camera, video or image can serve as a input to the application.

## Project Set Up and Installation
To run this applciation lcoally, one needs to have openVINO toolkit installed.

First enable the virtual environment: `source /opt/intel/openvino/bin/setupvars.sh`

Secondly, activate the python virtual environment. To install it, run the following commands in your terminal:

```
pip3 install virtualenv
virtualenv [name of the virtual environmant. For example, my_env]
source [my_env]/bin/activate
pip3 install -r ../requirements.txt
```

The requirements.txt file is located in the same directory as this `README.md`.
After all the testing has been done, simply deactivate the virtual environment by typing: `deactivate` .

Next step is to download the pretrained models. This can be done using the model downloader from openVINO:
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 --output outputs/
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 --output outputs/
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 --output outputs/
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --output outputs/
```


## Demo
To run a demo, navigate to the src directory and run the following command:
```
python3 main.py -fd_m ../../intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001 \
                -hp_m ../../intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 \
                -fl_m ../../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 \
                -ge_m ../../intel/gaze-estimation-adas-0002/INT8/gaze-estimation-adas-0002 \
                --cpu_extension /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so \
                -i ../bin/demo.mp4
```
## Documentation
The `main.py` takes the following as arguments:

* -fd_m or --facedetecionmodel: Path to face detection model's xml file with a trained model
* -hp_m or --headposeestimationmodel: Path to head pose estimation model's xml file with a trained model
* -fl_m or --faciallandmarksdetectionmodel: Path to facial landmarks detection model's xml file with a trained model
* -ge_m or --gazeestimationnmodel: Path to gaze estimation model's xml file with a trained model
* -i or --input: Path to image (IMG) or video (VID) or camera (CAM)
* -l or --cpu_extension: MKLDNN (CPU)-targeted custom layers; absolute path to a shared library with the kernels implementation
* -d or --device: Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD is acceptable (CPU by default)
* -pt or --prob_threshold: Probability threshold for detection's filtering (0.5 by default)
* -vis or --visualise: Visualise outputs: head (HEAD), eyes (EYES), gaze (GAZE) (no visualisation by default)

## Benchmarks
In the following, we have excluded any form of preprocessing and postprocessing and only measured the loading time and the inference time.
For the inference time, we measured inference time for each frame and saved those in the corresponding textual files in the folder outputs. Using this information, we calculated the statistics.

The results were obtained on Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz (further specifications can be found here: https://ark.intel.com/content/www/us/en/ark/products/97456/intel-core-i5-7300hq-processor-6m-cache-up-to-3-50-ghz.html).

### Face detecion model (face-detection-adas-binary-0001)
This model has only one precision: INT1.

Face detection      | INT1
---                 | --- 
model loading time  | 0.20340847969055176

The average inference time and its standard deviation:

Face detection      | INT1
---                 | --- 
average inference   | 0.017875400640197
standard deviation  | 0.004298714064644

### Head-pose estimation (head-pose-estimation-adas-0001)
This model has two precisions: FP16 and FP32.

Head-pose           | FP16                  | FP32
---                 | ---                   | ---
model loading time  | 0.05253171920776367   | 0.0377805233001709

The average inference time and its standard deviation:

Head-pose           | FP16              | FP32
---                 | ---               | ---
average inference   | 0.001249555814064 | 0.001970719482939
standard deviation  | 0.000202425200191 | 0.000542162237833

### Facial landmarks detection (landmarks-regression-retail-0009)
This model has two precisions: FP16 and FP32.

Facial landmarks    | FP16                  | FP32
---                 | ---                   | ---
model loading time  | 0.026652812957763672  | 0.026395320892333984

The average inference time and its standard deviation:

Facial landmarks    | FP16              | FP32
---                 | ---               | ---
average inference   | 0.000537637936867 | 0.00072748782271
standard deviation  | 0.000094775151847 | 0.000210170281644

### Gaze estimation (gaze-estimation-adas-0002)
This model has three percisions: INT8, FP16 and FP32.

Gaze estimation     | INT8                  | FP16                  | FP32
---                 | ---                   | ---                   | ---
model loading time  | 0.0691375732421875    | 0.06715250015258789    | 0.05340099334716797

The average inference time and its standard deviation:

Gaze estimation     | INT8              | FP16              | FP32
---                 | ---               | ---               | ---
average inference   | 0.001117447675285 | 0.001493251548623 | 0.002620688939499
standard deviation  | 0.000193803304417 | 0.000170847022527 | 0.000482894350576


## Results
As expected, by decreasing the precision, we will decrease the inference time. This is because we are performing faster calculations and more data can be stored in the faster memories.

Visual inspection showed no loss of performance. However, this may be due to the simplicity of the application.
Namely, face detection model is at the start of the pipeline and it is only available in one precision, so the next two models (facial landmarks detection and head-pose estimation) have the same start for both available precisions (FP16 and FP32). Since eyes are most likely one of the features by which our face detection model detects faces, the change of accuracy did not hamper the detection of eyes. There was no sign of worsened performance when using the lower precision for the head-pose estimation model.

In the end, the gaze estimation model takes detected eyes and head-pose angles to estimate the gaze vector, and it seems again that the performance was not hampered by using lower precision weights.

Due to the lack of the ground truth data, it is hard to give a definitive answer just by visual inspection.

In the end, the most plausible recommendation woudl be to use FP16 and INT8 precisions where choice is possible.

On the other hand, the load time for the model does not behave similarly: the difference is either negligible or it was faster for higehr precision (as in head-pose estimation).

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.
No async inference has been attempted. (Note: after we have run the first model, the following two can be run in parallel since there is no inter-dependance between their outputs.)

### Edge Cases
1) If there is no face detected in the frame, we will just skip the frame and continue with the next one.

2) If there is more than one face detected in the frame, we will take the one with the highest probability. A side-effect of this is that we can have alternating face with the highest probability and that would imply alternating the results in the subsequent steps of our application. (Note: it might be possible to use a tracker to track the face which we have chosen among other faces present in the frame, but this seems to be out of the scope of this project and nanodegree.)
