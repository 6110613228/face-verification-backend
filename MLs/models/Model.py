from abc import ABC, abstractmethod


class Skel(ABC):

    @abstractmethod
    def predict(image):
        pass

    def test(self):
        return 'test'
