# YoloV5 Lite Inference Logic

This code contains the logic to run the enrollment and inference part in the cloud. I have basically two models:
1. YOLOv5x-cls (front and side classification model to select images for enrolling)
2. EfficientNetB2 (for enrolling and inferncing)
The code will load the models from the artifacts folder and apply it to the images (and) videos in the "data/raw/" folder for enroll/inference.
The output files gets saved in the "data/processed" folder in the following format:

          data
            |---processed
              |---enroll_data
                |---house1
                |---house2
                |---enroll_metadata_db
                |---enroll_pets_db
              |---inference_data
                |---house1
                |---house2
                |---infernce_metadata_db
                |---inference_pets_db

The YoloV5x cls model was trained to classify whether a frame belongs to front facing cat or side facing cat.
The EfficientNetB2 model will exacts and saves the embedding from the images while enrolling. And match the embedding and do predictions while inferncing.

**NOTE**

    * Ensure that the models artifacts are present in the artifacts folder in the respective formats.
    * Place the files on which you want to inference on, in the `data/raw` folder under each households.

## Input parameters for yolo model

* weights : Path of the model artifact
* input_dir : Path of the input folder where the input files are located
* output_dir : Path of the output folder where the output files are to be saved
* enroll : if videos/images are to be used for pet enrollment. (True/False)
* augment : To apply augmentations on the images for enrolling. (True./False)


**NOTE**

     * To enable enroll and augment make changes in the `production/conf/yolo_conf.py` file.

## Input data

* Some sample input files have been provided in the `data/raw` folder.
* The input file can be an image or a video.(recommended video's for each pets under corresponding households)

## Output

#### While Enrolling
* While `enroll` = True, will prepare enroll images in `enroll_images` folder under corresponding pets.And save `enrolled_pets_db.xlsx` in data/processed/enroll_data path.
* While `enroll` and `augment` = True, will prepare enroll images in `enroll_with_aug` folder under corresponding pets.And save `enrolled_pets_db.xlsx` in data/processed/enroll_data path.

#### While Inferencing
* Excel file containing the pred_pet_type (cat/dog), pred_pet_id(pet name)(where "-1" means `Don't know` class),res_org (pet name predicted by lr model)  and the probability score from the lr model.


## Getting started

1. Ensure you have Miniconda installed and can be run from your shell. If not, download the installer for your platform from https://docs.conda.io/en/latest/miniconda.html

2. Open Anaconda Prompt and switch to the project root folder (i.e. the folder path containing this file)

3. Create a virtual environment with any name you like, for e.g., yolo_inference
```
(base):~/<proj-folder>$ conda create --name cloud_inference python==3.8.8
```

4. Activate the virtual environment
```
(base):~/<proj-folder>$ conda activate cloud_inference
```

5. Install the requirements
```
(yolo_inference):~/<proj-folder>$ pip install -r requirements.txt
```

6. Go to `production\notebooks\` and run `data_pipeline check.ipynb` script which runs the inference logic on the input files.