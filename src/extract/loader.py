from abc import ABC, abstractmethod
import kagglehub
from roboflow import Roboflow


class DataLoader(ABC):
    """
        Abstract base class for data loaders. The use of data loader is to load the latest 
        dataset from the source, which can be Roboflow, Kaggle, or any other source.
    """

    @abstractmethod
    def ingest(self):
    
        pass

    
class RoboflowDataLoader(DataLoader):
    def __init__(self,  **kwargs):
        self.dataset_location = kwargs["dataset_location"] # rawcoded
        self.api_key = kwargs["api_key"] # env
        self.workspace = kwargs["workspace"] # config
        self.project = kwargs["project"] # config
        self.dataset_name = kwargs["dataset_name"] # config
        self.version = kwargs["version"] # config
    def ingest(self):
        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(self.workspace).project(self.project)
        dataset = project.version(self.version).download(model_format = "yolov8", location = self.dataset_location, overwrite = True)
        self.dataset = dataset
        
class KaggleDataLoader(DataLoader):
    def __init__(self,  **kwargs):
        self.dataset_location = kwargs["dataset_location"]
        self.dataset_name = kwargs["dataset_name"]
    def ingest(self):
        path = kagglehub.dataset_download(self.dataset_name, output_dir=self.dataset_location, force_download=True)
        self.dataset = path
