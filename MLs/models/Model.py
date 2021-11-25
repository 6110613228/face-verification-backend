from abc import ABC, abstractmethod


class Skel(ABC):

    @abstractmethod
    def face_verification(image) -> bool:
        """
        @input array of 2 image
        @return true indicate that they are the same person
                false indicate that ther are NOT the same person
        """
        pass

    @abstractmethod
    def face_registration():
        pass

    def test(self):
        return 'test'
