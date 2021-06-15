import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset
from visualization import visualize_class_distribution


def augment_pixels(px, IMG_SIZE = 48):
    image = np.array(px.split(' ')).reshape(IMG_SIZE, IMG_SIZE).astype('float32')
    image = tf.image.random_flip_left_right(image.reshape(IMG_SIZE,IMG_SIZE,1))
    # Pad image size
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 12, IMG_SIZE + 12)
    # Random crop
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 1])
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.1, upper=0.4)
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0, 255)
    augmented = image.numpy().reshape(1, IMG_SIZE, IMG_SIZE)
    str_augmented = ' '.join(augmented.reshape(IMG_SIZE * IMG_SIZE).astype('int').astype(str))
    return str_augmented


def augment_data(df):
    print('augment data per class')
    # Calculate samples to get to same amount in each class
    valcounts = df.emotion.value_counts()
    valcounts_diff = valcounts[valcounts.idxmax()] - valcounts
    for emotion_idx, aug_count in valcounts_diff.iteritems():
        sampled = df.query("emotion==@emotion_idx").sample(aug_count, replace=True)
        sampled['pixels'] = sampled.pixels.apply(augment_pixels)
        df = pd.concat([df, sampled])
        print(emotion_idx, aug_count)

    # show new dataset
    print('----augmented dataset created----')
    print(df.shape)
    print(df.columns)
    print('--division--\n', df.Usage.value_counts())
    print('--classes--\n', df.emotion.value_counts())
    df.to_csv('archive/augmented_dataset.csv')
    return df


def preprocess_data():
    df = pd.read_csv("archive/fer2013/fer2013/fer2013.csv")

    print(df.shape)
    print('--division\n', df.Usage.value_counts())
    print('--classes\n', df.emotion.value_counts())
    lookup = ('anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral')

    visualize_class_distribution(df, lookup, title='initial')
    # augment data
    aug = augment_data(df[df['Usage'] == 'Training'])
    visualize_class_distribution(aug, lookup, title='augmented_train')

    X_train = aug['pixels'].apply(lambda x: np.array(x.split()).reshape(1, 48, 48).astype('float32')) / 255.
    X_train = np.stack(X_train, axis=0)
    y_train = aug['emotion']

    # Create tensors and save them
    X_test = df[(df['Usage'] == 'PrivateTest') | (df['Usage'] == 'PublicTest')]['pixels'] \
        .apply(lambda x: np.array(x.split()).reshape(1, 48, 48).astype('float32')) / 255.
    X_test = np.stack(X_test, axis=0)
    y_test = df[(df['Usage'] == 'PrivateTest') | (df['Usage'] == 'PublicTest')]['emotion']

    tensor_x = torch.tensor(X_train)  # transform to torch tensor
    tensor_y = torch.tensor(y_train.values)

    tensor_x_test = torch.tensor(X_test)  # transform to torch tensor
    tensor_y_test = torch.tensor(y_test.values)

    torch.save(tensor_x, 'archive/tensor_x.pt')
    torch.save(tensor_y, 'archive/tensor_y.pt')
    torch.save(tensor_x_test, 'archive/tensor_x_test.pt')
    torch.save(tensor_y_test, 'archive/tensor_y_test.pt')


def load_dataset(device):
    train = TensorDataset(torch.load('archive/tensor_x.pt').to(device),
                          torch.load('archive/tensor_y.pt').to(device))  # create your datset
    test = TensorDataset(torch.load('archive/tensor_x_test.pt').to(device),
                         torch.load('archive/tensor_y_test.pt').to(device))  # create your datset
    return train, test
