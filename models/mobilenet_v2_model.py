from base.base_model import BaseModel
from keras.models import Model
from keras.layers import *
from keras import backend as K
from models.mobilenet_v2 import MobileNetV2


class MobileNetV2Model(BaseModel):
    def __init__(self, config):
        super(MobileNetV2Model, self).__init__(config)
        self.build_model()

    def build_model(self):
        print("build mobile_net_v2")
        mobile_v2_model = MobileNetV2(
            input_shape=(224, 224, 3), alpha=1.0, include_top=False,
            weights=None)
        x = mobile_v2_model.output
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(self.config.num_output,
                        use_bias=True, name='Logits_77')(x)

        self.model = Model(inputs=mobile_v2_model.inputs, outputs=outputs)
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
