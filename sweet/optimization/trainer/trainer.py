"""nn trainer factory"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np


class SimpleTrainer():
    def __init__(self, epochs, early_stopper=True):
        self.epochs = epochs
        self.early_stopper = early_stopper

    def train(self, nn, ds, loss):
        # tf_ds = tf.data.Dataset.from_tensor_slices(ds.data)
        nn.compile('adam', loss)

        # print(np.array(ds.data).shape)

        # nn.fit(x=np.array(ds.data), y=np.array(ds.data), batch_size=1)
        nn.fit(x=[ds.data, ds.kers], y=[ds.data], batch_size=1, epochs=self.epochs)
        # nn.fit(ds, batch_size=1)


class TrainerTFDS():
    def __init__(self, epochs, batch_size=32, lr=1e-4, early_stopper=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Adam(lr)
        self.early_stopper = early_stopper


    def train(self, nn, ds, loss):
        nn.compile(self.optimizer, loss)
        train, train_steps, val, val_steps = ds.get_datasets(self.batch_size)

        callbacks = [tf.keras.callbacks.TerminateOnNaN()]
        if self.early_stopper:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0,
                    patience=20,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True,
                )
            )

        return nn.fit(
            train,
            steps_per_epoch=train_steps,
            validation_data=val,
            validation_steps=val_steps,
            epochs=self.epochs,
            callbacks=callbacks,
        )


NAME2CLASS = {
    'SimpleTrainer': SimpleTrainer,
    'TrainerTFDS': TrainerTFDS,
}


class TrainerFactory():
    def make(self, config):
        config = config.copy()
        obj_type = config.pop('type')
        if obj_type in NAME2CLASS:
            return NAME2CLASS[obj_type](**config)
        else:
            raise NotImplementedError(f"Unexpected type {obj_type}")
