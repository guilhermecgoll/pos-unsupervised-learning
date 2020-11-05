
class Person:
    id = 0
    label = 0
    data = []

    def __init__(self, id: int, label: int, data):
        self.id = id
        self.label = label
        self.data = data

    def toString(self):
        return 'Person [id={}, label={}, data={}]'.format(self.id, self.label, self.data)