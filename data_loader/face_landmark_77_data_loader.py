import numpy as np
import cv2
from pathlib import Path
from base.base_data_loader import BaseDataLoader

class FaceLandmark77DataLoader(BaseDataLoader):
    filp_mapping = np.array(
        [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29,
         30, 35, 34, 33, 32, 31,
         45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61,
         60, 67, 66, 65,
         68, 70, 69, 72, 71, 74, 73, 76, 75])
    def __init__(self, config):
        super(FaceLandmark77DataLoader, self).__init__(config)
        with open(self.config.data_file_train, "r") as f:
            self.x_train = [line.strip() for line in f.readlines()]
        with open(self.config.data_file_test, "r")  as f:
            self.x_test = [line.strip() for line in f.readlines()]

    def get_steps(self, train=True):
        if train:
            return int(np.ceil(len(self.x_train) * 1.0 / self.config.batch_size))
        else:
            return int(np.ceil(len(self.x_test) * 1.0/ self.config.batch_size))


    def process_single_data(self, data_id, img_folder, anno_folder, train=True):
        img_name = Path(data_id)
        img_path = str(img_folder / img_name)
        anno_path = str(anno_folder / (img_name.stem + ".txt"))

        # read raw image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        height, width, _ = image.shape

        with open(anno_path, "r") as f:
            annotation_lines = f.readlines()
        # ignore useless lines
        annotation_lines = annotation_lines

        annotation = [[float(value[0].strip()),
                       float(value[1].strip())]
                      for value in [line.split(",") for line in annotation_lines[2:]]]
        annotation = np.array(annotation)

        # prepare bbox data
        bbox = [float(val.strip()) for val in annotation_lines[1].split(",")]

        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        bbox = [np.clip(val[0], 0, val[1]) for val in zip(bbox, [width, height] * 2)]

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
        y = self.normalize_annotation(annotation, bbox).flatten()
        # if y.shape[0] == 136:
        #     print(img_name, anno_path, y.shape, annotation.shape, len(annotation_lines))

        return x, y

    def data_generator(self, train=True):
        if train:
            x_source, image_folder, anno_folder = \
                self.x_train, Path(self.config.image_folder_train), Path(self.config.annotation_folder_train)
        else:
            x_source, image_folder, anno_folder = \
                self.x_test, Path(self.config.image_folder_test), Path(self.config.annotation_folder_test)

        while True:
            X, Y = [], []
            cnt = 0
            for x_id in x_source:
                x, y = self.process_single_data(
                    x_id, image_folder, anno_folder, train)
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
        annotation = annotation[FaceLandmark77DataLoader.filp_mapping]

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