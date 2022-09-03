import flwr as fl
import tensorflow as tf


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = self.create_model()

        (x_train, y_train), (x_test, y_test) = self.load_data()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

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


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())
