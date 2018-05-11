from data_loader.simple_mnist_data_loader import SimpleMnistDataLoader
from data_loader.face_landmark_77_data_loader import FaceLandmark77DataLoader
from models.simple_mnist_model import SimpleMnistModel
from models.mobilenet_v2_model import MobileNetV2Model
from trainers.simple_mnist_trainer import SimpleMnistModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


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
    create_dirs([config.tensorboard_log_dir, config.checkpoint_dir])

    print('Create the data generator.')
    if hasattr(config, "data_set"):
        if config.data_set == "face_data_77":
            data_loader = FaceLandmark77DataLoader(config)
        else:
            data_loader = SimpleMnistDataLoader(config)
    else:
        data_loader = SimpleMnistDataLoader(config)

    print('Create the model.')
    if hasattr(config, "model_name"):
        if config.model_name == "mobile_net":
            model = MobileNetV2Model(config)
        else:
            model = SimpleMnistModel(config)
    else:
        model = SimpleMnistModel(config)

    print('Create the trainer')
    trainer = SimpleMnistModelTrainer(model.model, data_loader, config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
