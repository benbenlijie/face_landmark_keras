from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras import backend as K



class SimpleMnistModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(SimpleMnistModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, '%s-{epoch:03d}-{val_nme:.5f}.hdf5' % self.config.exp_name),
                monitor=self.config.checkpoint_monitor,
                mode=self.config.checkpoint_mode,
                save_best_only=self.config.checkpoint_save_best_only,
                save_weights_only=self.config.checkpoint_save_weights_only,
                verbose=self.config.checkpoint_verbose,
            )
        )

        self.callbacks.append(
                TensorBoard(
                    log_dir=self.config.tensorboard_log_dir,
                    write_graph=self.config.tensorboard_write_graph,
                )
            )

        # self.callbacks.append(
        #     LearningRateScheduler(self.lr_scheduler)
        # )

        if hasattr(self.config,"comet_api_key"):
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
            experiment.disable_mp()
            experiment.log_multiple_params(self.config)
            self.callbacks.append(experiment.get_keras_callback())

    def train(self):
        history = self.model.fit_generator(
            generator=self.data.data_generator(True),
            steps_per_epoch=self.data.get_steps(True),
            epochs=self.config.num_epochs,
            verbose=self.config.verbose_training,
            callbacks=self.callbacks,
            validation_data=self.data.data_generator(False),
            validation_steps=self.data.get_steps(False),
            max_queue_size=self.config.batch_size * 20,
            workers=12,
            use_multiprocessing=True,
            shuffle=True,
        )

        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['nme'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_nme'])

    def lr_scheduler(self, epoch):
        total_epoch = self.config.num_epochs
        if epoch <= total_epoch * 0.5:
            lr = self.config.learning_rate
        elif epoch <= total_epoch * 0.8:
            lr = self.config.learning_rate * 0.1
        else:
            lr = self.config.learning_rate * 0.01

        return lr