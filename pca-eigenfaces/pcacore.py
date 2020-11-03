from person import Person

from os import listdir
from os.path import isfile, join

import numpy as np

class PcaCore:

    p = 10
    train = []
    test = []

    def start(self):        
        # pessoa = Person(3, 3, vis)
        path = ".\orl"
        self._loadDataset(path)

    def _loadDataset(self, path: str):
        print(path)
        _files = [f for f in listdir(path) if isfile(join(path, f))]
        people = []
        for (_file) in _files:
            people.append(self._toPerson(path, _file))
        print(people)
    
    def _toPerson(self, path: str, filename: str):
        _file = filename.replace(".jpg", "")
        _parts = _file.split(sep='_')
        person = Person(int(_parts[0]), int(_parts[1]), np.zeros((5, 5), np.float32))
		# person.setData(getImageData(fileName));
        return person


