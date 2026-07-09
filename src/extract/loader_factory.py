
from .loader import KaggleDataLoader, RoboflowDataLoader


class LoaderFactory:
    @staticmethod
    def create_loader(loader_type, **kwargs):
        if loader_type == "roboflow":
            return RoboflowDataLoader(**kwargs)
        elif loader_type == "kaggle":
            return KaggleDataLoader(**kwargs)
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")