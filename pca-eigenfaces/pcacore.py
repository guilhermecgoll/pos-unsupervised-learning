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
            truePositiveCount = 0
            trueNegativeCount = 0
            for person in self.test:
                testData = person.data
                label = [0]
                confidence = [0]
                reconstructionError = [0]
                model.predict(testData, label, confidence, reconstructionError)
                labelOK = label[0] == person.label

                if(reconstructionError[0] > self.MAX_REC):
                    print('NOTA A PERSON - Predicted label:', label[0], ', confidence:', confidence, ', reconstructedError:', reconstructionError[0], ', original label:',
                          person.label)
                    if labelOK is False:
                        trueNegativeCount += 1
                elif confidence[0] > self.MAX_DISTANCE:
                    print('UKNOWN PERSON (by distance) - Predicted label:', label[0], ', confidence:',
                          confidence[0], ', reconstructedError:', reconstructionError[0], ', original label:',
                          person.label)
                    if labelOK is False:
                        trueNegativeCount += 1
                elif reconstructionError[0] > 2400 and confidence[0] > 1800:
                    print('UKNOWN PERSON (by two factors) - Predicted label:', label[0], ', confidence:',
                          confidence[0], ', reconstructedError:', reconstructionError[0], ', original label:',
                          person.label)
                    if labelOK is False:
                        trueNegativeCount += 1
                elif labelOK is True:
                    truePositiveCount += 1
                else:
                    print('UKNOWN - Predicted label:', label[0], ', confidence:',
                          confidence[0], ', reconstructedError:', reconstructionError[0], ', original label:',
                          person.label)
                    if labelOK is False:
                        trueNegativeCount += 1

                if(person.label <= 40):
                    # definir um limiar de confiança/distância de confiança
                    if confidence[0] < self.minDistance:
                        self.minDistance = confidence[0]
                    if confidence[0] > self.maxDistance:
                        self.maxDistance = confidence[0]

                    self.meanDistance += confidence[0]

                    # definir um limiar de confiança/distância de confiança
                    if reconstructionError[0] < self.minRec:
                        self.minDistance = reconstructionError[0]
                    if reconstructionError[0] > self.maxRec:
                        self.maxRec = reconstructionError[0]

                    self.meanRec += reconstructionError[0]
            trues = trueNegativeCount + truePositiveCount
            accuracy = trues / len(self.test) * 100

            print('numComponents:{0}, Percentual de acerto:{1} ({2} de {2}){3}'.format(_startComps, accuracy,
                                                                                       truePositiveCount, len(self.test)))
            print('truePositiveCount:{0}, trueNegativesCount:{1}'.format(
                truePositiveCount, trueNegativeCount))

            print('minDistance:{0}, maxDistance:{1}, meanDistance: {2}'.format(self.minDistance, self.maxDistance,
                                                                               self.meanDistance / len(self.test)))
            print('minRec:{}, maxRec:{}, meanRec: {}'.format(
                  self.minRec, self.maxRec, self.meanRec / len(self.test)))
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
