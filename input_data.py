import tensorflow as tf
from os import listdir
import numpy as np

file_path = "data/VOC2012/SegmentationClass/"
batch_size = 1

def parse_image(filename):
    image_string = tf.read_file(file_path + filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    return image_resized

def get_filenames():
    filenames = listdir(file_path)
    return filenames

def input_data():
    filenames = get_filenames()
    print(len(filenames))
    train_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    train_dataset = train_dataset.shuffle(100).repeat()
    train_dataset = train_dataset.map(parse_image).batch(batch_size)
    return train_dataset.make_one_shot_iterator()

if __name__ == '__main__':
    iterator = input_data()
    images = iterator.get_next()
    print(images)

