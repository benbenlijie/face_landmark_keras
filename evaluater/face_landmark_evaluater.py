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

    def evaluate_on_batch(self):
        if not self.weights_loaded:
            return
        if hasattr(self.config, "test_image"):
            x = cv2.imread(self.config.test_image, cv2.IMREAD_COLOR)
            X = self.prepare_input_image(x)
            Y_true = None
        else:
            eval_data = self.data.data_generator(False)
            X, Y_true = next(eval_data)
        Y_pred = self.model.model.predict_on_batch(X)
        padding = 0
        if hasattr(self.config, "padding"):
            padding = self.config.padding
        for i in range(len(X)):
            x_input = (X[i] + 1.) * 128.
            y_pred = (Y_pred[i] + 1.) / 2. * self.config.input_width
            y_pred = np.reshape(y_pred, (len(y_pred) // 2, 2))

            if Y_true is not None:
                y_true = (Y_true[i] + 1.) / 2. * self.config.input_width
                y_true = np.reshape(y_true, (len(y_true) // 2, 2))
            else:
                y_true = None

            self.output_result(x_input, y_pred, y_true, i+1)

            y_pred = (Y_pred[i] + 1.) / 2.
            y_pred = np.reshape(y_pred, (len(y_pred) // 2, 2))
            y_pred = y_pred * (np.array(x.shape[1::-1]) + padding * 2) - [padding, padding]
            y_pred = y_pred.astype(np.int32)
            self.output_result(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), y_pred, None, (i+1)*100, 5)

    def evaluate(self):
        if not self.weights_loaded:
            return
        eval_data = self.data.data_generator(False)
        nme_array = []
        for i in range(self.data.get_steps(False)):
            X, Y_true = next(eval_data)
            Y_pred = self.model.model.predict_on_batch(X)
            Y_pred = (Y_pred + 1.) / 2. * self.config.input_width
            Y_pred = np.reshape(Y_pred, (Y_pred.shape[0], Y_pred.shape[1] // 2, 2))
            Y_true = (Y_true + 1.) / 2. * self.config.input_width
            Y_true = np.reshape(Y_true, (Y_true.shape[0], Y_true.shape[1] // 2, 2))
            nme_array.append(self.nme_result(Y_pred, Y_true))
            print(nme_array[-1])
        print("nme result: ", np.mean(np.array(nme_array)))

    def output_result(self, x, y_pred, y_true=None, i=0, circle_size=2):
        img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        cv2.imwrite("val_test/raw.jpg", img)
        for j in range(len(y_pred)):
            cv2.circle(img, tuple(y_pred[j]), circle_size, (255, 0, 0), circle_size)
            cv2.putText(img, str(j), tuple(y_pred[j]), cv2.FONT_HERSHEY_COMPLEX_SMALL, circle_size, (255, 0, 0), circle_size)
        name = "val_test/val_pred_{:04d}.jpg".format(i)
        print("write image: ", name)
        cv2.imwrite(name, img)
        if y_true is not None:
            for j in range(len(y_true)):
                cv2.circle(img, tuple(y_true[j]), circle_size, (0, 255, 0), circle_size)
            name = "val_test/val_test_{:04d}.jpg".format(i)
            cv2.imwrite(name, img)

    def prepare_input_image(self, x):
        padding = 0
        if hasattr(self.config, "padding"):
            padding = self.config.padding

        if x.shape[0] > self.config.input_height or x.shape[1] > self.config.input_width:
            x = cv2.copyMakeBorder(x, padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, (255, 255, 255))
            x = cv2.resize(x, (self.config.input_width, self.config.input_height))

        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x / 128. - 1.
        x = x.astype(np.float32)
        X = np.expand_dims(x, axis=0)
        return X

    def nme_result(self, y_predicts, y_trues):
        interocular_distance = np.sqrt(np.sum(
            (np.mean(y_trues[:, 36:42, :], axis=1) - np.mean(y_trues[:, 42:48, :], axis=1)) ** 2, axis=-1))
        return np.mean(np.sum(np.sqrt(np.sum((y_predicts - y_trues) ** 2, axis=-1)), axis=-1)) / \
               np.mean((interocular_distance * self.config.num_output / 2))

