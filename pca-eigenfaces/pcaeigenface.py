import numpy as np
import cv2 as cv
from math import sqrt


class PCAEigenFace:

    numComponents = 0
    mean = np.zeros((1, 1))
    diffs = np.zeros((1, 1))
    covariance = np.zeros((1, 1))
    eigenvalues = []
    eigenvectors = []
    labels = []
    projections = []

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
        destino = destino.reshape(80, 80).transpose()
        cv.imwrite(filename, destino)

    def _calcDiff(self, train: list):
        sample = train[0].data
        num_rows, num_cols = sample.shape
        self.diffs = np.zeros((num_rows, len(train)), dtype=sample.dtype)
        for i, j in enumerate(self.diffs):
            for k, l in enumerate(j):
                mv = self.mean[i, 0]
                data = train[k].data
                dv = data[i, 0]
                diff = dv - mv
                self.diffs[i, k] = diff

    def _calcCovariance(self, train: list):
        self.covariance = self._mul(self.diffs.transpose(), self.diffs)

    def _mul(self, matA, matB):
        num_rowsA, num_colsA = matA.shape
        num_rowsB, num_colsB = matB.shape
        d = np.zeros((num_rowsA, num_colsB), dtype=float)
        c = np.zeros((num_rowsA, num_colsB), dtype=float)

        cv.gemm(matA, matB, 1, d, 1, dst=c)
        return c

    def _calcEigen(self, train: list):
        retval, self.eigenvalues, self.eigenvectors = cv.eigen(
            self.covariance, eigenvalues=None, eigenvectors=None)

    def _calcEigenFaces(self, train: list):
        evt = self.eigenvectors.transpose()
        rows, cols = evt.shape
        components = cols
        if self.numComponents > 0:
            components = self.numComponents
        ev_k = np.copy(evt[:, 0:components])

        self.eigenFaces = self._mul(self.diffs, ev_k)
        rows, cols = self.eigenFaces.shape
        i = 0
        while i < cols:
            ef = self.eigenFaces[:, i: i + 1]
        # 	Normalização L2 = Yi = Xi / sqrt(sum((Xi)^2)), onde i = 0...rows-1
            cv.normalize(ef, ef)
            i += 1

    def _calcProjections(self, train: list):
        self.projections = np.zeros(
            (self.numComponents, len(train)), dtype=float)
        rows, cols = self.diffs.shape
        i = 0
        while i < cols:
            diff = self.diffs[:, i: i + 1]
            w = self._mul(self.eigenFaces.transpose(), diff)
            self.projections[:, i: i + 1] = w
            self.labels.insert(i, train[i].label)
            i += 1

    def predict(self, testData: list, label: list, confidence: list, reconstructionError: list):
        diff = np.zeros(())
        diff = cv.subtract(testData, self.mean, diff)

        # Calcula os pesos da imagem desconhecida.
        w = self._mul(self.eigenFaces.transpose(), diff)

        # Calcular o vizinho mais próximo dessa projeção 'desconhecida'
        minJ = 0
        minDistance = self._calcDistance(w, self.projections[:, minJ:1])
        j = 1
        rows, cols = self.projections.shape
        while j < cols:
            distance = self._calcDistance(w, self.projections[:, j: j + 1])
            if (distance < minDistance):
                minDistance = distance
                minJ = j
            j += 1

        label.insert(0, self.labels[minJ])
        confidence.insert(0, minDistance)

        reconstruction = self._calcReconstruction(w)
        reconstructionError.insert(0, cv.norm(
            testData, reconstruction, cv.NORM_L2))
        self._saveImage(testData, '.\itestData.jpg')
        self._saveImage(reconstruction, '.\ireconstruction.jpg')

    def _calcDistance(self, p, q):
        # Distância euclidiana:
        # d = sqrt(sum(pi - qi)^2)

        distance = 0
        i = 0
        rows, cols = p.shape

        while i < (rows - 1):
            pi = p[i, 0]
            qi = q[i, 0]
            d = pi - qi
            distance += d * d
            i += 1

        result = sqrt(distance)
        return result

    def _calcReconstruction(self, w):
        result = self._mul(self.eigenFaces, w)
        cv.add(result, self.mean, result)

        return result
