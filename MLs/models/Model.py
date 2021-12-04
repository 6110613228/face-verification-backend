from abc import ABC, abstractmethod


class Skel(ABC):

    @abstractmethod
    def face_verification(image) -> bool:
        """
        @param array of 2 images\n
        @return\n
            1. result 
                - true indicate that they are the same person
                - false indicate that ther are NOT the same person
        """
        pass

    @abstractmethod
    def face_registration():
        pass

    def test(self):
        return 'test'
