from trainer import Trainer
from model import ImageClassifier, build_model
from dataset import ImageGenerator
import os
import argparse
import numpy as np
import tensorflow as tf

cfg = tf.compat.v1.ConfigProto()
# cfg.gpu_options.per_process_gpu_memory_fraction = 0.8
cfg.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=cfg)


def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='dataset root path', required=True)
    parser.add_argument('--class_file', type=str, help='class file path', required=True)
    parser.add_argument('--ratio', type=float, help='train-validation split ratio', default=0.7)
    parser.add_argument('--bs', type=int, help='batch size', default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--size', type=int, help='input image size', default=224)
    parser.add_argument('--mode', type=str, help='Training mode', choices=['fit', 'eager'], default='fit')
    return parser.parse_args()


def get_compiled_model(model_train):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    outputs = model_train(inputs)
    return tf.keras.Model(inputs, outputs)


def main(config):
    data_gen = ImageGenerator(root_path=config.root,
                              size=config.size,
                              class_file=config.class_file,
                              batch_size=config.bs,
                              ratio=config.ratio)
    train_ds, val_ds = data_gen.set_train()

    model = ImageClassifier(out_dims=1)
    optimizer = tf.optimizers.Adam(learning_rate=config.lr)
    loss_fn = tf.losses.BinaryCrossentropy(from_logits=False)

    trainer = Trainer(model, optimizer, loss_fn)

    lowest_loss = tf.constant(np.inf)

    for epoch in range(config.epochs):
        train_loss = trainer.train_one_epoch(train_ds, config.bs)
        val_loss = trainer.val_one_epoch(val_ds, config.bs)
        if val_loss <= lowest_loss:
            lowest_loss = val_loss

        print(f"Epoch({epoch+1}/{config.epochs}): train_loss={float(train_loss)}  valid_loss={float(val_loss)}  loweset_loss={float(lowest_loss)}")


if __name__ == '__main__':
    config = define_argparser()
    main(config)
