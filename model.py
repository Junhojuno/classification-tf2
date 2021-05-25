import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential, layers


class ImageClassifier(tf.keras.Model):
    """Image Classifier based on CNN"""

    def __init__(self, out_dims=1):
        super(ImageClassifier, self).__init__()
        self.out_dims = out_dims
        self.feature_extractor = ResNet50(include_top=False,
                                          input_shape=(224, 224, 3))
        self.classifier = Sequential([
            layers.Conv2D(filters=512, kernel_size=3, padding='same'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(units=128, activation='relu'),
            layers.Dense(units=32, activation='relu'),
            layers.Dense(units=self.out_dims, activation='sigmoid')
        ])

    def call(self, x):
        x = self.feature_extractor(x)  # shape : (bs,7,7,2048)
        x = self.classifier(x)
        return x
