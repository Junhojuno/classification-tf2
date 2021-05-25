"""make tf.data.Dataset(Generator)"""
import tensorflow as tf
from preprocess import collect_data, transform_images, transform_labels

class ImageGenerator:
    """create a dataset generator"""
    def __init__(self, root_path, size, class_file, batch_size, ratio=0.7):
        self.root_path = root_path
        self.size = size
        self.class_table = self._set_class_table(class_file)
        self.batch_size = batch_size
        self.ratio = ratio
        
    def _set_class_table(self, class_file):
        text_initializer = tf.lookup.TextFileInitializer(filename=class_file,
                                                         key_dtype=tf.string,
                                                         key_index=0,
                                                         value_dtype=tf.int64,
                                                         value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
        class_table = tf.lookup.StaticHashTable(text_initializer, default_value=-1)
        return class_table
        
    def _get_files(self, root_path):
        """load dataset which has images/labels

        Returns:
            tf.data.Dataset: generator of (image_path, class_text)
        """
        return collect_data(root_path)
    
    def _train_val_split(self, ds, ratio):
        """train-validation set split"""
        n_total = tf.cast(tf.data.experimental.cardinality(ds), tf.float32)
        n_train = tf.cast(n_total * ratio, tf.int64)
        train_ds = ds.take(n_train)
        val_ds = ds.skip(n_train)
        return train_ds, val_ds
    
    def set_train(self):
        """Train Validation split & make tf.data.Dataset each"""
        n_data, ds = self._get_files(self.root_path)
        ds = ds.shuffle(n_data, reshuffle_each_iteration=False)
        train_ds, val_ds = self._train_val_split(ds, self.ratio)
        
        train_ds = train_ds.map(
            lambda x,y : (transform_images(x, self.size), 
                          transform_labels(y, self.class_table)
                          )
            )
        val_ds = val_ds.map(
            lambda x,y : (transform_images(x, self.size), 
                          transform_labels(y, self.class_table)
                          )
            )
        
        train_ds = train_ds.batch(self.batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.batch(self.batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_ds, val_ds
        
if __name__ == '__main__':
    root = '/home/juno/data/classification/cats_and_dogs/training_set/training_set'
    dataset = ImageGenerator(root_path=root,
                             size=224,
                             class_file='./class.names',
                             batch_size=1)
    train_ds, val_ds = dataset.set_train()
    print(f'number of train data : {tf.data.experimental.cardinality(train_ds)}')
    for image, label in train_ds:
        print(image.shape)
    print(f'number of train data : {tf.data.experimental.cardinality(val_ds)}')
    for image, label in val_ds.take(5):
        print(image.shape)