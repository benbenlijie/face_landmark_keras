from base.base_evaluater import BaseEvaluater
import cv2
import numpy as np


class FaceLandmarkEvaluater(BaseEvaluater):
    def __init__(self, model, data, config):
        super(FaceLandmarkEvaluater, self).__init__(model, data, config)
        self.weights_loaded = False
        if hasattr(self.config, "best_checkpoint"):
            self.model.load(self.config.best_checkpoint)
            self.weights_loaded = True
        else:
            print("Do you forget to set the 'best_checkpoint' in config file")
            return

    def evaluate(self):
        if not self.weights_loaded:
            return
        if hasattr(self.config, "test_image"):
            x = cv2.imread(self.config.test_image, cv2.IMREAD_COLOR)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = x / 128. - 1.
            x = x.astype(np.float32)
            X = np.expand_dims(x, axis=0)
            Y_true = None
        else:
            eval_data = self.data.data_generator(False)
            X, Y_true = next(eval_data)
        Y_pred = self.model.model.predict_on_batch(X)
        for i in range(len(X)):
            x = (X[i] + 1.) * 128.
            y_pred = (Y_pred[i] + 1.) / 2. * self.config.input_width
            y_pred = np.reshape(y_pred, (len(y_pred) // 2, 2))

            if Y_true is not None:
                y_true = (Y_true[i] + 1.) / 2. * self.config.input_width
                y_true = np.reshape(y_true, (len(y_true) // 2, 2))
            else:
                y_true = None

            self.output_result(x, y_pred, y_true, i)

    def output_result(self, x, y_pred, y_true=None, i=0):
        img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        cv2.imwrite("val_test/raw.jpg", img)
        for j in range(len(y_pred)):
            cv2.circle(img, tuple(y_pred[j]), 2, (255, 0, 0), 2)
        name = "val_test/val_pred_{:03d}.jpg".format(i + 1)
        print("write image: ", name)
        cv2.imwrite(name, img)
        if y_true is not None:
            for j in range(len(y_true)):
                cv2.circle(img, tuple(y_true[j]), 2, (0, 255, 0), 2)
            name = "val_test/val_test_{:03d}.jpg".format(i + 1)
            cv2.imwrite(name, img)