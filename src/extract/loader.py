from abstract import ABC, abstractmethod
from config import GEGE_API_KEY, workspace1, project1, dataset_name
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
    def __init__(self, dataset_location, api_key, workspace, project, dataset_name, version):
        self.dataset_location = dataset_location
        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.dataset_name = dataset_name
        self.version = version
    def ingest(self):
        rf = Roboflow(api_key=GEGE_API_KEY)
        project = rf.workspace(self.workspace).project(self.project)
        dataset = project.version(self.version).download(model_format = "yolov8", location = self.dataset_location, overwrite = True)
        self.dataset = dataset
        
class KaggleDataLoader(DataLoader):
    def __init__(self, dataset_location, api_key, dataset_name):
        self.dataset_location = dataset_location
        self.api_key = api_key
        self.dataset_name = dataset_name
    def ingest(self):
        # api = KaggleApi()
        # api.authenticate()
        # api.dataset_download_files(self.dataset_name, path=self.dataset_location, unzip=True)
        # self.dataset = self.dataset_location
        pass # Kaggle API is not working, so I will skip this part for now.