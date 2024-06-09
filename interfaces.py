from abc import ABC, abstractmethod
from typing import List, Tuple
import glm

class WorldInterface(ABC):
    @abstractmethod
    def get_surfaces(self) -> List[Tuple[glm.vec3, glm.vec3, glm.vec3]]:
        pass

class PlayerInterface(ABC):
    @property
    @abstractmethod
    def model(self) -> WorldInterface:
        pass

    @property
    @abstractmethod
    def velocity(self) -> glm.vec3:
        pass

    @velocity.setter
    @abstractmethod
    def velocity(self, value: glm.vec3):
        pass

    @property
    @abstractmethod
    def position(self) -> glm.vec3:
        pass

    @position.setter
    @abstractmethod
    def position(self, value: glm.vec3):
        pass
