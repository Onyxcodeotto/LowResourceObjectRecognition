from abc import ABC, abstractmethod

class Task(ABC):
    """
    A singular transformer in silver pipeline.

    Args:
        ABC (_type_): _description_
    """
    def __init__(self, name=None):
        self.name = name if name else self.__class__.__name__
    @abstractmethod
    def run(self, **upstream_outputs):

        pass
    @staticmethod
    def single_input(upstream_outputs):
        """Convenience for tasks with exactly one upstream dependency."""
        if len(upstream_outputs) != 1:
            raise ValueError(
                f"Expected exactly one upstream input, got {list(upstream_outputs)}"
            )
        return next(iter(upstream_outputs.values()))