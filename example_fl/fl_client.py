import flwr as fl
import tensorflow as tf
from pydantic import BaseModel
from itertools import islice
import numpy as np


class FitConfig(BaseModel):
    epochs: int = 1
    batch_size: int = 32


class LoadConfig(BaseModel):
    func: str = "tf.keras.datasets.cifar10.load_data"
    args: list = []
    kwargs: dict = {}


class TransformConfig(BaseModel):
    slice: list = None  # islice([], 1,1,1)


class Configs(BaseModel):
    server_address: str = "127.0.0.1:8080"
    fit: FitConfig = FitConfig()
    load: LoadConfig = LoadConfig()
    transform: TransformConfig = TransformConfig()


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = self.create_model()

        (x_train, y_train), (x_test, y_test) = self.load_data()
        # numpy.ndarray
        """
        [
            [
                [ 59  62  63]
                [ 43  46  45]
                [ 50  48  43]
                [158 132 108]
                [152 125 102]
                [148 124 103]
            ]
            [
                [ 25  24  21]
                [ 16   7   0]
                [ 49  27   8]
                [118  84  50]
                [120  84  50]
                [109  73  42]
            ]
        ]
        """

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # 一般的にテストデータは30%などで分割される
        print(
            f"""
        x_train={len(self.x_train)}
        y_train={len(self.y_train)}
        x_test={len(self.x_test)}
        y_test={len(self.y_test)}
        """
        )

    def create_model(self):
        model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def load_data(self):
        return tf.keras.datasets.cifar10.load_data()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

    @classmethod
    def run(cls, configs: Configs = None):
        """Start Flower client"""
        configs = configs or Configs()
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=cls())


CifarClient.run()
