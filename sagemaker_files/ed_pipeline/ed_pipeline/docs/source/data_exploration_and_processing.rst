===============================
Data Exploration and Processing
===============================


1. Data Sources
================
The data used for training the models were collected from the various sources as mentioned below:

1.1 Training Data
--------------------


+---------------------------------------------------------------------------------------+----------------+-----------------------+--------------+
| Source                                                                                | Pet types      |          #images      |Used for      |
|                                                                                       |                +-------+-------+-------+training      |
|                                                                                       |                | cat   | dog   | other |              |
+=======================================================================================+================+=======+=======+=======+==============+
| `Petfinder <https://www.petfinder.com/>`_ *****                                       | cats, dogs     |79,321 |65,022 |   0   |EfficientNetB2|
|                                                                                       |                |       |       |       |              |
+---------------------------------------------------------------------------------------+----------------+-------+-------+-------+--------------+
| `IIIT Oxford <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_                          | cats, dogs     | 1,188 | 2,498 |   0   |              |
+---------------------------------------------------------------------------------------+----------------+-------+-------+-------+              |
| `Kaggle pet faces <https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection>`_ | cats, dogs     | 1,188 | 2,498 |   0   |              |
+---------------------------------------------------------------------------------------+----------------+-------+-------+-------+              |
| `Kaggle cat keypoints data <https://www.kaggle.com/datasets/crawford/cat-dataset>`_   | cats           | 9,997 |   0   |   0   |              |
+---------------------------------------------------------------------------------------+----------------+-------+-------+-------+              |
| `Stanford <https://vision.stanford.edu/aditya86/ImageNetDogs/>`_                      | dogs           | 0     |11,059 |    0  |              |
+---------------------------------------------------------------------------------------+----------------+-------+-------+-------+YOLOv5        +
|                                                                                       |others (bird,   | 0     |   0   |17,304 |              |
|                                                                                       |hamster,lizard, |       |       |       |              |
| `Open Images dataset <https://storage.googleapis.com/openimages/web/index.html>`_     |mouse, rabbit,  |       |       |       |              |
|                                                                                       |tortoise etc.)  |       |       |       |              |
+---------------------------------------------------------------------------------------+----------------+-------+-------+-------+              |
| `MIT indoor data <https://web.mit.edu/torralba/www/indoor.html>`_                     |others          |    0  |   0   | 2,497 |              |
|                                                                                       |(indoor scenes) |       |       |       |              |
+---------------------------------------------------------------------------------------+----------------+-------+-------+-------+--------------+


.. note::
    The other category refers the pets which are not cats/dogs and belong to other common pet categories such as hamsters, rabbits, birds etc.
    These images were used to train YOLO model to reject pet classes other than cat and dog.


*For the petfinder data, `Petfinder API <https://www.petfinder.com/developers/v2/docs/>`_ was used to scrape images and corresponding metadata.


1.2 Pipeline Data
-------------------------
The petfinder data was used to test the pet recognition pipeline.
As specified in the table it contains 79,321 images belongs to 14,534 cats and  65,022 images belongs to 16,630 dogs.


- **Artificial Household Creation**

In order to evaluate the model at the household level, we have assigned the pets from the selected data to different artificial/dummy households.


- **Train Test Split**

Each pet images were split into train and test with the default split ratio of 60:40

Following is the distribution of pipeline data by pet type.

+-------------+--------------+---------------+--------------+----------+--------------+
| Pet Type    |Pet Per House |#train-images  | #test-images | #pets    |  #households |
+=============+==============+===============+==============+==========+==============+
| cats        |  2           |   23,725      |  15,708      |  7,214   |   3,607      |
|             +--------------+---------------+--------------+----------+--------------+
|             |  3           |   10,586      |   6,987      |  3,234   |   1,078      |
|             +--------------+---------------+--------------+----------+--------------+
|             |  4+          |   13,448      |   8,867      |  4,086   |   1,021      |
+-------------+--------------+---------------+--------------+----------+--------------+
| Total cat                  |   47,759      |  31,562      | 14,534   |   5,706      |
+-------------+--------------+---------------+--------------+----------+--------------+
|      dogs   |  2           |   19,256      |  12,950      |  8,230   |   4,115      |
|             +--------------+---------------+--------------+----------+--------------+
|             |  3           |    8,667      |   5,788      |  3,732   |   1,244      |
|             +--------------+---------------+--------------+----------+--------------+
|             |  4+          |   10,987      |   7,374      |  4,668   |   1,167      |
+-------------+--------------+---------------+--------------+----------+--------------+
| Total dog                  |   38,910      |  26,112      |   16,630 |   6,526      |
+----------------------------+---------------+--------------+----------+--------------+
| **Total**                  |**86,669**     |**57,674**    |**31,164**| **6,528**    |
+----------------------------+---------------+--------------+----------+--------------+

