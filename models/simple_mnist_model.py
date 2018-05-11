from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import *
from keras import backend as K



class SimpleMnistModel(BaseModel):
    def __init__(self, config):
        super(SimpleMnistModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(96, (7, 2), name="conv_1", input_shape=(224, 224, 3)))
        self.model.add(Activation("relu", name="relu_1"))
        self.model.add(MaxPool2D(3, 3, name="max_pool_1"))
        self.model.add(Dropout(0.5, name="dropout_1"))

        self.model.add(Conv2D(256, 5, name="conv_2"))
        self.model.add(Activation("relu", name="relu_2"))
        self.model.add(MaxPool2D(2, 2, name="max_pool_2"))
        self.model.add(Dropout(0.5, name="dropout_2"))

        self.model.add(Conv2D(512, 3, padding="same", name="conv_3"))
        self.model.add(Activation("relu", name="relu_3"))
        self.model.add(Conv2D(512, 3, padding="same", name="conv_4"))
        self.model.add(Activation("relu", name="relu_4"))
        self.model.add(Conv2D(512, 3, padding="same", name="conv_5"))
        self.model.add(Activation("relu", name="relu_5"))
        self.model.add(MaxPool2D(3, 3, name="max_pool_5"))
        self.model.add(Dropout(0.5, name="dropout_5"))

        self.model.add(Flatten(name="flatten"))
        self.model.add(Dense(4096, name="dense_6"))
        self.model.add(Activation("relu", name="relu_6"))
        self.model.add(Dropout(0.5, name="dropout_6"))

        self.model.add(Dense(4096, name="dense_7"))
        self.model.add(Activation("relu", name="relu_7"))
        self.model.add(Dropout(0.5, name="dropout_7"))

        self.model.add(Dense(self.config.num_output, name="output_77"))

        if hasattr(self.config, "weights"):
            print("loading weights from", self.config.weights)
            self.model.load_weights(self.config.weights, by_name=True)

        self.model.compile(
            loss=self.face_loss,
            optimizer=self.config.optimizer,
            metrics=[self.nme],
        )
        # print(self.model.metrics_names)

    def face_loss(self, y_true, y_pred):
        y_pred_reshape = K.reshape(y_pred, (-1, self.config.num_output//2, 2))
        y_true_reshape = K.reshape(y_true, (-1, self.config.num_output//2, 2))
        interocular_distance = K.sqrt(K.sum(
                (K.mean(y_true_reshape[:, 36:42, :], axis=1) - K.mean(y_true_reshape[:, 42:48, :], axis=1)) ** 2, axis=-1))
        return K.mean(K.mean(
            K.sqrt(K.sum(
                (y_pred_reshape - y_true_reshape) ** 2, axis=-1)), axis=-1) / interocular_distance)

    def nme(self, y_true, y_pred):
        y_pred_reshape = K.reshape(y_pred, (-1, self.config.num_output//2, 2))
        y_true_reshape = K.reshape(y_true, (-1, self.config.num_output//2, 2))
        interocular_distance = K.sqrt(K.sum(
            (K.mean(y_true_reshape[:, 36:42, :], axis=1) - K.mean(y_true_reshape[:, 42:48, :], axis=1)) ** 2, axis=-1))
        return K.mean(K.sum(
            K.sqrt(K.sum((y_pred_reshape - y_true_reshape) ** 2, axis=-1))
            , axis=-1)) / K.mean((interocular_distance * self.config.num_output / 2))
        # return K.mean(K.sum(
        #     K.sqrt(K.sum((y_pred_reshape - y_true_reshape) ** 2, axis=-1))
        #     , axis=-1))
