from abc import ABC, abstractmethod

class Pipe(ABC):
    """
    Generic class for pipeline. This class will run
    pipeline once. 
    Note:
    Will change to other method like airflow
    when automation onto the data layer is needed.
    If this is still used, that means automation doesnt reach data orchestration layer yet.
    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def run(self, config):
        """
        Run the pipeline once.
        """
        pass