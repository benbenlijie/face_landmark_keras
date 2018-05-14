from data_loader.simple_mnist_data_loader import SimpleMnistDataLoader
from models.simple_mnist_model import SimpleMnistModel
from data_loader.face_landmark_77_data_loader import FaceLandmark77DataLoader
from models.mobilenet_v2_model import MobileNetV2Model
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
import tensorflow as tf
from keras import backend as K
from pathlib import Path


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

    if hasattr(config, "best_checkpoint"):
        model.load(config.best_checkpoint)
        frozen_graph = freeze_session(K.get_session(),
                                      output_names=[out.op.name for out in model.model.outputs])

        ckpt_path = Path(config.best_checkpoint)
        tf.train.write_graph(frozen_graph, str(ckpt_path.parent), ckpt_path.with_suffix(".pb").name, as_text=False)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

if __name__ == '__main__':
    main()
