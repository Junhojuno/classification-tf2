import tensorflow as tf


class Trainer:
    """Train & validation procedure per each step"""

    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, image_bs, label_bs):
        """training step"""
        with tf.GradientTape() as tape:
            logits = self.model(image_bs)
            step_loss = self.loss_fn(label_bs, logits)
            # step_loss += sum(self.model.losses)
        gradients = tape.gradient(step_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return step_loss

    @tf.function
    def val_step(self, image_bs, label_bs):
        """validation step"""
        logits = self.model(image_bs)
        val_step_loss = self.loss_fn(label_bs, logits)
        return val_step_loss
    
    def train_one_epoch(self, train_ds, batch_size):
        train_epoch_loss = 0.0
        num_train_batches = 0.0
        for image, label in train_ds:
            loss_value = self.train_step(image, label)
            mean_loss = tf.math.divide(loss_value, tf.cast(batch_size, tf.float32))
            train_epoch_loss += mean_loss
            num_train_batches += 1
        return train_epoch_loss / num_train_batches

    def val_one_epoch(self, val_ds, batch_size):
        val_epoch_loss = 0.0
        num_val_batches = 0.0
        for image, label in val_ds:
            loss_value = self.val_step(image, label)
            mean_loss = tf.math.divide(loss_value, tf.cast(batch_size, tf.float32))
            val_epoch_loss += mean_loss
            num_val_batches += 1
        return val_epoch_loss / num_val_batches
