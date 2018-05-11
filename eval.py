from data_loader.simple_mnist_data_loader import SimpleMnistDataLoader
from models.simple_mnist_model import SimpleMnistModel
from trainers.simple_mnist_trainer import SimpleMnistModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
import cv2
import numpy as np

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.tensorboard_log_dir, config.checkpoint_dir, "val_test"])

    print('Create the data generator.')
    data_loader = SimpleMnistDataLoader(config)

    print('Create the model.')
    model = SimpleMnistModel(config)

    if hasattr(config, "best_checkpoint"):
        model.load(config.best_checkpoint)
    else:
        print("Do you forget to set the 'best_checkpoint' in config file")
        return
    eval_data = data_loader.data_generator(False)
    X, Y_true = next(eval_data)
    Y_pred = model.model.predict_on_batch(X)
    for i in range(len(X)):
        x = (X[i] + 1.) * 128.
        img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        y_pred = (Y_pred[i] + 1.) / 2. * config.input_width
        y_true = (Y_true[i] + 1.) / 2. * config.input_width
        y_pred = np.reshape(y_pred, (len(y_pred) // 2, 2))
        y_true = np.reshape(y_true, (len(y_true) // 2, 2))

        for j in range(len(y_pred)):
            cv2.circle(img, tuple(y_pred[j]), 2, (255, 0, 0), 2)
        cv2.imwrite("val_test/val_pred_{:03d}.jpg".format(i+1), img)
        for j in range(len(y_pred)):
            cv2.circle(img, tuple(y_true[j]), 2, (0, 255, 0), 2)
        cv2.imwrite("val_test/val_test_{:03d}.jpg".format(i + 1), img)
    # predict = model.model.predict_generator(
    #     data_loader.data_generator(False),
    #     steps=data_loader.get_steps(False),
    #     workers=4)
    # print(predict.shape)


if __name__ == '__main__':
    main()
