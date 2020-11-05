import numpy as np

class PCAEigenFace:

    numComponents = 0
    mean = np.zeros((1,1))

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
        data = train[0].data
        print(data.shape)
        num_rows, num_cols, wtf = data.shape
        self.mean = np.zeros((num_rows, num_cols))

        for person in train:
            _data = person.data
            i = 0
            while i < num_rows:
                mv = self.mean[i,0:][0]
                pv = _data[i,0:][0]
                mv += pv
                #descobrir como atualizar a matriz
                self.mean[i,0:] = mv

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
