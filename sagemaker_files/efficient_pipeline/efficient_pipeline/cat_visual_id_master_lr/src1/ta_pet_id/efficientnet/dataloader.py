import cv2
import numpy as np
import os
import random
import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


class PetDataset(tf.keras.utils.Sequence):
    def __init__(self, base_path, image_paths, augumentation, batch_size, IMG_SIZE, train=True):
        self.base_path = base_path
        self.image_paths = image_paths
        self.augumentation = augumentation
        self.batch_size = batch_size
        self.IMG_SIZE = IMG_SIZE
        self.train = train  # helpful in reshuffling ...
        self.on_epoch_end()

    # MUST DEFINE
    def __len__(self):  # number of batches in a sequence ...
        return len(self.image_paths) // self.batch_size

        # If you want to modify your dataset between epochs you may implement on_epoch_end

    # we can shuffle data for every batch :)
    def on_epoch_end(self):

        self.indices = np.arange(len(self.image_paths))  # resets indices ot normal order after epoch end :)
        if self.train:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        indices = self.indices[self.batch_size * idx: self.batch_size * (idx + 1)]  # full batch at index idx
        batch_of_anchor_images, batch_of_positive_images, batch_of_negative_images = self.__generate_data(indices)
        return (batch_of_anchor_images, batch_of_positive_images, batch_of_negative_images), np.zeros(
            (self.batch_size, 1))

    def __generate_data(self, indices):
        batch_of_anchor_images = np.empty((self.batch_size, self.IMG_SIZE, self.IMG_SIZE, 3), 'uint8')
        batch_of_positive_images = np.empty((self.batch_size, self.IMG_SIZE, self.IMG_SIZE, 3), 'uint8')
        batch_of_negative_images = np.empty((self.batch_size, self.IMG_SIZE, self.IMG_SIZE, 3), 'uint8')
        for i in range(len(indices)):
            anchor_index = random.randint(0, len(self.image_paths[indices[i]]) - 1)
            positive_index = random.randint(0, len(self.image_paths[indices[i]]) - 1)
            while positive_index == anchor_index:
                positive_index = random.randint(0, len(self.image_paths[indices[i]]) - 1)
            indices_copy = list(indices[:i]) + list(indices[i + 1:])
            negative_cat_index = random.choice(indices_copy)
            negative_index = random.randint(0, len(self.image_paths[negative_cat_index]) - 1)
            # Anchor Img
            img_path = os.path.join(self.base_path, self.image_paths[indices[i]][anchor_index][0])
            face_data = self.image_paths[indices[i]][anchor_index][1]  # 3,4,1,2
            anchor_image = cv2.imread(img_path)
            anchor_image = anchor_image[int(face_data[2]):int(face_data[3]), int(face_data[0]):int(face_data[1]), :]
            anchor_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2RGB)

            # Positive img
            img_path = os.path.join(self.base_path, self.image_paths[indices[i]][positive_index][0])
            face_data = self.image_paths[indices[i]][positive_index][1]
            positive_image = cv2.imread(img_path)
            positive_image = positive_image[int(face_data[2]):int(face_data[3]), int(face_data[0]):int(face_data[1]), :]
            positive_image = cv2.cvtColor(positive_image, cv2.COLOR_BGR2RGB)

            # Negative Img
            img_path = os.path.join(self.base_path, self.image_paths[negative_cat_index][negative_index][0])
            face_data = self.image_paths[negative_cat_index][negative_index][1]
            negative_image = cv2.imread(img_path)
            negative_image = negative_image[int(face_data[2]):int(face_data[3]), int(face_data[0]):int(face_data[1]), :]
            negative_image = cv2.cvtColor(negative_image, cv2.COLOR_BGR2RGB)

            augmented_anchor = cv2.resize(anchor_image, (self.IMG_SIZE, self.IMG_SIZE),
                                          interpolation=cv2.INTER_LANCZOS4)
            augmented_positive = cv2.resize(positive_image, (self.IMG_SIZE, self.IMG_SIZE),
                                            interpolation=cv2.INTER_LANCZOS4)
            augmented_negative = cv2.resize(negative_image, (self.IMG_SIZE, self.IMG_SIZE),
                                            interpolation=cv2.INTER_LANCZOS4)

            batch_of_anchor_images[i] = augmented_anchor
            batch_of_positive_images[i] = augmented_positive
            batch_of_negative_images[i] = augmented_negative

        return batch_of_anchor_images, batch_of_positive_images, batch_of_negative_images


def process_train_data(context, train_meta):
    """
    Function to filter the train data.

    Parameters
    ----------
    context : object
        context object to locate config parameters

    Returns
    -------
    object
        object for train and test dataset

    """

    train_meta['face_loc'] = train_meta['face_loc'].fillna(0)
    train_meta['to_use'] = np.where((train_meta.face_loc == 0), 0, 1)
    imgs_per_pet_dict = train_meta[train_meta['to_use'] == 1].groupby('house_pet_id').size().to_dict()
    train_meta['imgs_per_pet'] = train_meta['house_pet_id'].apply(
        lambda x: imgs_per_pet_dict[x] if x in imgs_per_pet_dict.keys()
        else 0)
    train_meta['to_use'] = np.where((train_meta.imgs_per_pet < 2) | (train_meta.face_loc == 0), 0, 1)

    split_ratio = context.efficientnet['training']['test_size']
    pets = list(train_meta[train_meta['to_use'] == 1]['house_pet_id'].values)
    SPLIT_POINT = int(split_ratio * len(pets))
    train_pets = pets[SPLIT_POINT:]
    test_pets = pets[:SPLIT_POINT]
    train_meta['sample_type'] = ""
    train_meta['sample_type'] = np.where(train_meta['house_pet_id'].isin(train_pets), "train",
                                         train_meta['sample_type'])
    train_meta['sample_type'] = np.where(train_meta['house_pet_id'].isin(test_pets), "val", train_meta['sample_type'])
    train_meta['sample_type'] = np.where(train_meta['to_use'] == 0, "", train_meta['sample_type'])
    return train_meta


def train_test_split(context, train_meta):
    """
    Function will return train train and test set after cleaning.

    Parameters
    ----------
    context : object
        context object to locate config parameters

    Returns
    -------
    object
        object for train and test dataset

    """
    train_meta = train_meta[train_meta['to_use'] == 1].reset_index(drop=True)
    train_meta['for_triplet'] = list(zip(train_meta['image_path'], train_meta['face_loc']))
    train_pets = train_meta[train_meta['sample_type'] == 'train'].groupby(['house_id', 'pet_id'])['for_triplet'].apply(
        list).values
    test_pets = train_meta[train_meta['sample_type'] == 'val'].groupby(['house_id', 'pet_id'])['for_triplet'].apply(
        list).values
    base_path = context.data_catalog['efficientnet']['raw_folder']
    BATCH_SIZE = context.efficientnet['training']['batch_size']
    IMG_SIZE = 256
    with mirrored_strategy.scope():
        train_dataset = PetDataset(base_path, train_pets, None, BATCH_SIZE, IMG_SIZE)

        test_dataset = PetDataset(base_path, test_pets, None, BATCH_SIZE, IMG_SIZE,
                                  train=False)
    return train_dataset, test_dataset
