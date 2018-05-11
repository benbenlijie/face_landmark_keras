from base.base_data_loader import BaseDataLoader
import numpy as np
import cv2
from pathlib import Path


class SimpleMnistDataLoader(BaseDataLoader):
    filp_mapping = np.array(
        [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29,
         30, 35, 34, 33, 32, 31,
         45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61,
         60, 67, 66, 65])
    def __init__(self, config):
        super(SimpleMnistDataLoader, self).__init__(config)
        with open(self.config.data_file_train, "r") as f:
            self.x_train = [line.strip() for line in f.readlines()]
        with open(self.config.data_file_test, "r")  as f:
            self.x_test = [line.strip() for line in f.readlines()]

        with open(self.config.bbox_file_train, "r") as f:
            bbox_lines = [line.strip().split(",") for line in f.readlines()[1:]]
            self.bbox_train = {line[0]: list(map(int, line[1:])) for line in bbox_lines}

        with open(self.config.bbox_file_test, "r") as f:
            bbox_lines = [line.strip().split(",") for line in f.readlines()[1:]]
            self.bbox_test = {line[0]: list(map(int, line[1:])) for line in bbox_lines}

        # (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        # self.X_train = self.X_train.reshape((-1, 28 * 28))
        # self.X_test = self.X_test.reshape((-1, 28 * 28))

    def get_steps(self, train=True):
        if train:
            return int(np.ceil(len(self.x_train) * 1.0 / self.config.batch_size))
        else:
            return int(np.ceil(len(self.x_test) * 1.0/ self.config.batch_size))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def process_single_data(self, data_id, bbox, img_folder, anno_folder, train=True):
        img_name = Path(data_id)
        img_path = str(img_folder / img_name)
        anno_path = str(anno_folder / (img_name.stem + ".pts"))

        # read raw image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        height, width, _ = image.shape

        # prepare bbox data
        bbox = [np.clip(val[0], 0, val[1]) for val in zip(bbox, [width, height] * 2)]

        with open(anno_path, "r") as f:
            annotation_lines = f.readlines()
        # ignore useless lines
        annotation_lines = annotation_lines[3:-1]
        annotation = [[float(value[0]),
                       float(value[1])]
                      for value in [line.split(" ") for line in annotation_lines]]

        annotation = np.array(annotation)

        if train:
            # enhance image with random flip and random rotate
            # randomFlip image+bbox+annotation
            image, bbox, annotation = self._randomFlip(image, bbox, annotation)
            # randomRotate image+bbox+annotation
            image, bbox, annotation = self._randomRotate(image, bbox, annotation)

        # crop face
        bbox = self.adjust_bbox(bbox, [width, height])
        xmin, ymin, xmax, ymax = bbox
        face = image[ymin:ymax, xmin:xmax]
        originWidth, originHeight = xmax - xmin, ymax - ymin

        x_scale, y_scale = 1. / originWidth, 1. / originHeight
        try:
            face = cv2.resize(face, (self.config.input_width, self.config.input_height))
        except:
            print("error!!!", img_name, train, face.shape, bbox)

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face / 128. - 1.
        x = face.astype(np.float32)

        bbox = np.array(bbox, np.float32)
        scale = np.array([x_scale, y_scale], np.float32)
        annotation = np.array(annotation, np.float32)
        # adjust annotation and normalized to -1~1 value
        y = self.normalize_annotation(annotation, bbox)
        return x, y.flatten()

    def data_generator(self, train=True):
        if train:
            x_source, bbox_source, image_folder, anno_folder = \
                self.x_train, self.bbox_train, Path(self.config.image_folder_train), Path(self.config.annotation_folder_train)
        else:
            x_source, bbox_source, image_folder, anno_folder = \
                self.x_test, self.bbox_test, Path(self.config.image_folder_test), Path(self.config.annotation_folder_test)

        while True:
            X, Y = [], []
            cnt = 0
            for x_id in x_source:
                x, y = self.process_single_data(
                    x_id, bbox_source[x_id], image_folder, anno_folder, train)
                X.append(x)
                Y.append(y)
                cnt += 1
                if cnt == self.config.batch_size:
                    cnt = 0
                    yield (np.array(X), np.array(Y))
                    X, Y = [], []

    def normalize_annotation(self, annotation, bbox):
        xmin, ymin, xmax, ymax = bbox
        originWidth, originHeight = xmax - xmin, ymax - ymin
        return np.array([[(point[0] - xmin) * 2. / originWidth - 1., (point[1] - ymin) * 2. / originHeight - 1.]
                               for point in annotation], dtype=np.float32)

    def _randomFlip(self, img, bbox, annotation):
        """

        :param img: [height, width, channel]
        :param bbox:  [xmin, ymin, xmax, ymax]
        :param annotation: [[x, y], ...] shape == [64, 2]
        :return:
        img, bbox, annotation
        """
        # 50% to flip
        if np.random.random() < 0.5:
            return img, bbox, annotation
        height, width, _ = img.shape
        # flip img
        img = cv2.flip(img, 1)
        # flip bbox
        xmin, ymin, xmax, ymax = bbox
        xmin, xmax = np.clip(width - xmax, 0, width - 1), np.clip(width - xmin, 0, width - 1)
        bbox = [xmin, ymin, xmax, ymax]
        # flip annotation
        annotation = np.array([[width - point[0], point[1]] for point in annotation])
        # reorder point
        annotation = annotation[SimpleMnistDataLoader.filp_mapping]

        return img, bbox, annotation

    def _randomRotate(self, img, bbox, annotation, angle=15):
        """

        :param img: [height, width, channel]
        :param bbox:  [xmin, ymin, xmax, ymax]
        :param annotation: [[x, y], ...] shape == [64, 2]
        :return:
        img, bbox, annotation
        """
        # 33% to rotate
        if np.random.random() < 2 / 3:
            return img, bbox, annotation
        elif np.random.random() < 0.5:
            angle *= -1
        height, width, channel = img.shape
        rotationMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

        # rotate img
        avg_color_per_row = np.average(img, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        newImg = cv2.warpAffine(img, rotationMat, (width, height), borderValue=avg_color)

        # rotate bbox
        bboxPoints = np.array([[x, y, 1] for x in bbox[::2] for y in bbox[1::2]])
        bboxPoints = rotationMat.dot(bboxPoints.T).T
        newBbox = [np.min(bboxPoints[:, 0]), np.min(bboxPoints[:, 1]), np.max(bboxPoints[:, 0]),
                   np.max(bboxPoints[:, 1])]
        newBbox = [int(np.clip(pos[0], 0, pos[1])) for pos in zip(newBbox, [width, height] * 2)]

        xmin, ymin, xmax, ymax = newBbox
        originWidth, originHeight = xmax - xmin, ymax - ymin
        if originWidth == 0 or originHeight == 0:
            return img, bbox, annotation

        # rotate annotation
        annotationMat = np.concatenate((annotation, np.ones((len(annotation), 1))), axis=1)
        annotation = rotationMat.dot(annotationMat.T).T

        return newImg, newBbox, annotation

    def adjust_bbox(self, bbox, imgSize):
        xmin, ymin, xmax, ymax = bbox
        width, height = xmax - xmin, ymax - ymin
        newBbox = [int(np.clip(pos[0] + pos[1] * pos[2], 0, pos[3] - 1))
                   for pos in zip(bbox, self.config.margin, [width, height] * 2, imgSize * 2)]

        return newBbox