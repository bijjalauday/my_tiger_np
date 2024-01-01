===============
Experiments
===============

Below is the brief list of experiments conducted during POC Phase-1 and Phase-2


Pet Recognition based on Image matching
""""""""""""""""""""""""""""""""""""""""""""""""""""



1. Image based on distance similarity

    - Computing  distance between images using various distance measures (pixel by pixel)
    - Tried multiple similarity distances i.e. Euclidean, Cosine, RMSE etc.

2. Image based on histogram similarity

    - Used Histogram of Gradients(HOG) to calculate features from image.
    - Comparing the HOG features between images ( computed by counting the # of occurrences of gradient orientation in localized portions of an image.)



Pet Type Classification
""""""""""""""""""""""""""""""""""""""""""""""""""""

3. Cat vs No-Cat classification models

    - Using different models to classify cat and non-cat images:
        - ResNet
        - InceptionV3
        - MobileNet
        - YOLOv5


4. Cat's profile classification

    - Used YOLOv5 to detect eye locations and that is used to classify Cat's face profile whether it is Front-Facing or Side-Facing.
    - The profile was further used to evaluate the cat vs non-cat performance.

Pet Face Detection
""""""""""""""""""""""""""""""""""""""""""""""""""""

5. Using Haar Cascade

    - Used OpenCv haar cascades to detect and crop cat face.


6. Using Keras Keypoint model

    - Trained Keras keypoint model to detect 9 keypoints on the cat's face.
    - These keypoints covers Cat's eyes, nose and ears.


Pet Type Classification & Face Detection
""""""""""""""""""""""""""""""""""""""""""""""""""""

7. YOLOv5 for cat/non cat classification and face detection

    - Trained YOLO to classify cat/non-cat as well as to detect cat face in image.


8. YOLOv5 retrained on augmented images

    - In order to handle various realistic scenarios, we augmented the original images.

    - Trained YOLOv5 model on images with various augmentations like:

        - Blurring the image
        - Adding gaussian noise to image
        - Image brightness variation
        - Varying Hue and Saturation values of the image.

9. YOLOv5 for both cat and dog

    - The YOLOv5 model gets extended to detect both cat and dog.
    - Trained single YOLOv5 model to classify cat/dog/other as well as detect cat/dog face in the image.



10. YOLOv5 retrained with more other common pets

    - In order to reduce the misclassification as recognizing other common pets as cat and dog, these common pets (rabbit, hamster etc) are added to the training data.

    - Trained YOLOv5 models to recognize these common pets as other class.

Embedding Model
""""""""""""""""""""""""""""""""""""""""""""""""

11. Following models were developed to extract embeddings from the pet’s face.

    - MobileNetV2 - for cat
    - EfficientNetB2 - for cat
    - EfficientNetB2 - for dog
    - EfficientNetB2 - single model for both cat and dog


Other Experiments
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

12. Blur detection in image

    - Categorize the image as blur or non-blur by using Fourier Fast Transform(FFT) method.
    - Based on this approach we can detect whether camera is out of focus or if there any smudge on lens by processing image feeds.





