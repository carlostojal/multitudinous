from abc import ABC, abstractmethod

# abstract configuration class
class Config(ABC):

    @abstractmethod
    def parse_from_file(self, filename: str) -> None:
        pass
