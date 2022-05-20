from abc import ABC, abstractmethod


class Wavefront(ABC):

    @abstractmethod
    def coeffs(self):
        pass

    @abstractmethod
    def coeffs_dict(self):
        pass

    def update(self, view_dist=None):
        if view_dist:
            assert view_dist > 0, f"Incorrect viewing distance = {view_dist}"
        self._viewing_dist = view_dist

    @abstractmethod
    def calc(self, x, y):
        pass
