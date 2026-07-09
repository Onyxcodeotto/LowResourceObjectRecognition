from .pipe import Pipe
from ..extract.loader_factory import LoaderFactory


class BronzePipeline(Pipe):
    def __init__(self):
        self.factory = LoaderFactory()
    def run(self, config):

        for dataset in config["datasets"]:
            print("Api Key:", dataset.get("api_key", "No API Key Provided"))
            self.factory.create_loader(loader_type = dataset["dataset_type"], **dataset["params"]).ingest()