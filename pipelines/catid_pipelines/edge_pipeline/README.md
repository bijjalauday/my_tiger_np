# YoloV5 Lite Inference Logic

This code contains the logic to load the yolov5 nano tflite model from the artifacts folder and apply it to the images (and) videos in the "data/raw/" folder for enroll/inference. The output files gets saved in the "data/processed" folder in the following format:

          data
            |---processed
              |---enroll_data
                |---house1
                  |---cat
                    |---cat1
                      |---video1
                        |---video1_1.jpg
                        |---video1_2.jpg
                      |---video2
                        |---video2_1.jpg
                        |---video2_2.jpg
                      |---top_frames
                        |---video1_1.jpg
                        |---video1_2.jpg
                        |---video2_1.jpg
                        |---video2_2.jpg
                      |---predictions.json
                      |---top_predictions.xlsx
                    |---cat2
                  |---dog
              |---inference
                |---video1
                  |---video1_1.jpeg
                  |---video1_2.jpeg
                  |---video1_3.jpeg
                  |---video1_5.jpeg
                  |---video1_10.jpeg
                |---image1
                  |---image1.jpg
                |---predictions.json
                |---top_frames
                  |---video1_1.jpg
                  |---image1.jpg
                  |---video1_2.jpeg
                  |---video1_3.jpeg
                  |---video1_10.jpeg
                |---top_frames.xlsx


The YoloV5 nano model was re-trained on 100187 images with  100 + 70 (incrimental) epochs. The model was trained to detect whether a frame contains a cat or a dog.

The model input is an image/video and following are the outputs:
          1. Bounding box co-ordinates of the detected faces(center of face, width and height of face).
          2. Box confidence score - probability that bounding box contains the classified object.
          3. Type of detected pet (cat or dog in our case).

**NOTE**

    * Ensure that the YoloV5 nano model artifact is present in the artifacts folder in the format of tflite.
    * Place the files on which you want to inference on, in the `data/raw` folder
    * The pipeline saves all the frames which detects a cat having confidence greater than the confidence threshold. An output json file also gets saved which contains information about the frames which detects cat/dog with confidence higher than the confidence threshold.


## Input parameters

* weights : Path of the model artifact
* input_dir : Path of the input folder where the input files are located
* output_dir : Path of the output folder where the output files are to be saved
* conf_thresh : Threshold to reject frames less than the desired confidence value
* top_frames : Number of top frames to be saved for recognition part (to be send to cloud)
* enroll : if videos/images are to be used for pet enrollment.

**NOTE**

     * The `confidence` threshold is pre-set as `0.6` in the `conf.py` file.
     * The `top_frames` is pre-set as `5` in the `conf.py` file.
     * The `enroll` is pre-set as `False` in the `conf.py` file

## Input data

* Some sample input files have been provided in the `data/raw` folder.
* The input file can be an image or a video.

## Output

* JSON file containing the bounding-box coordinates, class and the confidence score for frames having a confidence score higher than the confidence threshold
* Frames (Images) having a confidence score higher than the confidence threshold.

## Getting started

1. Ensure you have Miniconda installed and can be run from your shell. If not, download the installer for your platform from https://docs.conda.io/en/latest/miniconda.html

2. Open Anaconda Prompt and switch to the project root folder (i.e. the folder path containing this file)

3. Create a virtual environment with any name you like, for e.g., yolo_inference
```
(base):~/<proj-folder>$ conda create --name yolo_edge_inference python==3.8.8
```

4. Activate the virtual environment
```
(base):~/<proj-folder>$ conda activate yolo_inference
```

5. Install the requirements
```
(yolo_inference):~/<proj-folder>$ pip install -r requirements.txt
```

6. Run the `main.py` script which runs the inference logic on the input files
```
(yolo_inference):~/<proj-folder>$ python main.py
```