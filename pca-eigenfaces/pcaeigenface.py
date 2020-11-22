import numpy as np
import cv2 as cv


class PCAEigenFace:

    numComponents = 0
    mean = np.zeros((1, 1))

    def __init__(self, numComponents: int):
        self.numComponents = numComponents

    def train(self, train: list):
        self._calcMean(train)
        self._calcDiff(train)
        self._calcCovariance(train)
        self._calcEigen(train)
        self._calcEigenFaces(train)
        self._calcProjections(train)

    def _calcMean(self, train: list):
        sample = train[0].data
        num_rows, num_cols = sample.shape
        self.mean = np.zeros((num_rows, num_cols))

        for person in train:
            _data = person.data
            i = 0
            while i < num_rows:
                matriz_valores = self.mean[i, 0]
                person_valores = _data[i, 0]
                nova_soma = (matriz_valores + person_valores)
                self.mean[i, 0] = nova_soma
                i += 1

        i = 0
        while i < num_rows:
            matriz_valores = self.mean[i, 0]
            matriz_valores /= len(train)
            self.mean[i, 0] = matriz_valores
            i += 1

        self._saveImage(self.mean, '.\imagemMedia.jpg')

    def _saveImage(self, image, filename: str):
        destino = np.zeros((image.shape))
        # Para salvar a imagem é necessário normalizar na escala 64Bits entre 0 e 255 (8 bits)
        cv.normalize(image, destino, 0, 255, cv.NORM_MINMAX, cv.CV_64FC1)
        print(image.dtype)
        print(destino.dtype)
        destino = destino.reshape(80, 80).transpose()
        cv.imwrite(filename, destino)

    def _calcDiff(self, train: list):
        print('called _calcDiff')

    def _calcCovariance(self, train: list):
        print('called _calcCovariance')

    def _calcEigen(self, train: list):
        print('called _calcEigen')

    def _calcEigenFaces(self, train: list):
        print('called _calcEigenFaces')

    def _calcProjections(self, train: list):
        print('called _calcProjections')
