==================
Project Overview
==================



1. Objective
===================
A deep learning pipeline to identify a particular pet (dog/cat) among multiple pets in a household based on the images/video feed from an IoT device.


2. Approach
==================

Pet recognition is a 2 step process:

- Face Detection

- Face Recognition


2.1 Face Detection

    - Detects the pet’s face in the pet image.

    - Classify the type of pet as dog or cat.

    .. image:: images/project_overview/face_detection_flow.png

2.2 Face Recognition

    - Extract the embeddings from the pet’s face.

    - Compare it against other enrolled pets’ embeddings in the same household.

    - Predict the Pet ID based on the embedding closest to the input image.

    .. image:: images/project_overview/face_recognition_only_flow.png

.. note::
    Enrollment is the process of adding a new pet to the system.
    This is covered in the following section: :ref:`Enrollment`





