"""Name -> Task class map used by the config-driven flow assembler.

Only tasks that are implemented and importable are registered. Add entries here
as new task classes land (ExtractPlate, ImageMetadataExtraction,
SourceMetadataAttachment, AssignSplit, Merge, GenerateOCRManifest, ...).
"""
from src.transforms.tasks.load_task import (
    LoadCSVToText,
    LoadYOLOToBoxes,
    LoadYOLOToText,
)
from src.transforms.tasks.validate import BboxSanityCheck
from src.transforms.tasks.merge import Merge
from src.transforms.tasks.output import GenerateYOLO

TASK_REGISTRY = {
    "YOLO2Boxes": LoadYOLOToBoxes,
    "YOLO2Text": LoadYOLOToText,
    "CSV2Text": LoadCSVToText,
    "BboxSanityCheck": BboxSanityCheck,
    "Merge": Merge,
    "GenerateYOLO": GenerateYOLO,
}
