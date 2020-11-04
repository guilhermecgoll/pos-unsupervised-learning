from person import Person

from os import listdir
from os.path import isfile, join

import numpy as np
import cv2 as cv

class PcaCore:

    p = 10
    train = []
    test = []

    def start(self):        
        # pessoa = Person(3, 3, vis)
        path = ".\orl"
        self._loadDataset(path)

    def _sortFunc(self, e):
        return e.id

    def _loadDataset(self, path: str):
        _files = [f for f in listdir(path) if isfile(join(path, f))]
        people = []
        for (_file) in _files:
            people.append(self._toPerson(path, _file))
        people = people.sort(key=self._sortFunc)
    
    
    def _toPerson(self, path: str, filename: str):
        _file = filename.replace(".jpg", "")
        _parts = _file.split(sep='_')
        _image = self._getImageData(join(path, filename))
        person = Person(int(_parts[0]), int(_parts[1]), _image)
        return person

    def _getImageData(self, filename: str):
        _img = cv.imread(filename)
        _resizedImg = cv.resize(_img, (80, 80))
        return _resizedImg
