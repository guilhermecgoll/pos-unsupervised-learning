from person import Person
from pcaeigenface import PCAEigenFace

import sys
from os import listdir
from os.path import isfile, join
import random

import numpy as np
import cv2 as cv
from skimage.util import img_as_float


class PcaCore:

    trainSize = 7  # holdout 70/30
    train = []
    test = []

    startComps = 15
    maxComps = 15

    minDistance = sys.float_info.max
    maxDistance = sys.float_info.min
    meanDistance = 0

    minRec = sys.float_info.max
    maxRec = sys.float_info.min
    meanRec = 0

    MAX_DISTANCE = 2500
    MAX_REC = 2900

    def start(self):
        path = ".\orl"
        self._loadDataset(path, self.trainSize)
        _startComps = self.startComps
        while _startComps <= self.maxComps:
            model = PCAEigenFace(_startComps)
            model.train(self.train)
            _startComps += 1

    def _sortFunc(self, e):
        return e.id

    def _loadDataset(self, path: str, trainSize: int):
        _files = [f for f in listdir(path) if isfile(join(path, f))]
        people = []
        for (_file) in _files:
            people.append(self._toPerson(path, _file))
        people.sort(key=self._sortFunc)

        numSamplesPerPerson = 10
        personSamples = []
        for person in people:
            personSamples.append(person)
            if len(personSamples) == numSamplesPerPerson:
                while len(personSamples) > trainSize:
                    index = random.randint(0, len(personSamples) - 1)
                    self.test.append(personSamples[index])
                    personSamples.pop(index)

                if trainSize == numSamplesPerPerson:
                    self.test.extend(personSamples)

                self.train.extend(personSamples)
                personSamples.clear()

    def _toPerson(self, path: str, filename: str):
        _file = filename.replace(".jpg", "")
        _parts = _file.split(sep='_')
        _image = self._getImageData(join(path, filename))
        person = Person(int(_parts[0]), int(_parts[1]), _image)
        return person

    def _getImageData(self, filename: str):
        _img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        _resizedImg = cv.resize(_img, (80, 80))
        # converte num vetor coluna de 6400 x 1
        _resizedImg = _resizedImg.transpose().reshape(-1, 1)
        # converte a imagem de 8 bits em 64bits
        novaImagem = img_as_float(_resizedImg)
        return novaImagem

    def _showImg(self, img):
        cv.imshow('dst_rt', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
