"""Preprocessing image data"""
import os
import tensorflow as tf


def collect_data(root_path):
    """collect image path & label"""
    image_path_ds = tf.data.Dataset.list_files(os.path.join(root_path, '*/*.jpg'))
    label_ds = image_path_ds.map(lambda x: tf.strings.split(x, sep='/')[-2])
    n_data = tf.data.experimental.cardinality(label_ds)
    final_ds = tf.data.Dataset.zip((image_path_ds, label_ds))
    return n_data, final_ds


def transform_images(image_path, size):
    """transform image from path to Tensor"""
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.
    image = tf.image.resize(image, (size, size))
    return image


def transform_labels(class_text, class_to_label):
    """transform label from string to integer"""
    label = class_to_label.lookup(class_text)
    return label


if __name__ == '__main__':
    ds = tf.data.Dataset.list_files(os.path.join('/home/juno/data/classification/cats_and_dogs/training_set/training_set', '*/*.jpg'))
    print(ds)
