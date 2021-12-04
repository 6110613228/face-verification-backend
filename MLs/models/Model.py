from abc import ABC, abstractmethod


class Skel(ABC):

    @abstractmethod
    def face_verification(images) -> bool:
        """
        @param Array of `EXACT` 2 images\n
        @return\n
            1. Boolean
                - true indicate that they are the same person
                - false indicate that ther are NOT the same person
        """
        pass

    @abstractmethod
    def face_recognition(images: list) -> list:
        """
        @param Array of images\n
        @return Array of recognized classe(s) `in order` to each image in input's array
        """
        pass

    @abstractmethod
    def face_registration():
        pass

    def test(self):
        return 'test'
