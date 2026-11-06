
from src.transforms.tasks.task import Task
import os 
import pandas as pd
import yaml
"""To be implemented

### Input processing
1. YOLO to text
2. Text Read
3. Image metadata extraction
4. Image preprocessing (resize, crop, etc.)

### Mid process
1. Classify plate color
2. Classify daytime or nighttime



### Output Generation
1. Generate YOLO # For vehicle detection task
2. Generate OCR Manifest

"""


def _load_class_names(yaml_file):
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)["names"]
    
def _parse_label_file(label_path, class_names, ignore_classes):
    """Read a YOLO label file into ``[(cx, cy, w, h, class_name), ...]``.

    Each line is ``class_id cx cy w h`` in normalised coords. ``class_names`` may
    be a list (index = id) or a dict ({id: name}). Boxes whose class name is in
    ``ignore_classes`` are skipped.
    """
    def _name(cls_id):
        if isinstance(class_names, dict):
            return class_names.get(cls_id, str(cls_id))
        if 0 <= cls_id < len(class_names):
            return class_names[cls_id]
        return str(cls_id)

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            cls_name = _name(cls_id)
            if cls_name in ignore_classes:
                continue
            boxes.append((cx, cy, w, h, cls_name))
    return boxes
    
class LoadYOLOToText(Task):
    """
    Extract YOLO detection of character into text/string format.
    """
    def __init__(self, image_dir, yaml_file, labels_dir, split="train", ignore_classes = None, name=None):
        super().__init__(name)
        self.image_dir = image_dir
        self.yaml_file = yaml_file
        self.labels_dir = labels_dir
        self.split = split
        self.ignore_classes = set(ignore_classes or [])


    def run(self, **upstream_outputs):
        # Implementation for extracting YOLO detection to text
        """
        Config Requirements
        1. Image dir
        2. YAML file
        3. Labels dir
        
        Upstream Requireents
        None
        
        Output:
        Dataframe
        id|file_path|text
        """
        if not os.path.isdir(self.image_dir):
            return pd.DataFrame(columns = ["id", "file_path", "text", "boxes", "split"])
        class_names = _load_class_names(self.yaml_file)

        rows = []
        for image_filename in sorted(os.listdir(self.image_dir)):
            stem, ext = os.path.splitext(image_filename)
            if ext.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            
            label_path = os.path.join(self.labels_dir, f"{stem}.txt")
            if not os.path.isfile(label_path):
                continue
            boxes = _parse_label_file(label_path, class_names, self.ignore_classes)
            if not boxes:
                continue
            
            boxes = sorted(boxes, key=lambda b:b[0])
            text = "".join(b[4] for b in boxes)
            
            rows.append({
                "id": stem,
                "file_path": os.path.join(self.image_dir, image_filename),
                "text": text,
                "boxes": boxes,
                "split": self.split,
            })
        return pd.DataFrame(rows, columns=["id", "file_path", "text", "boxes", "split"])
        
    
class LoadCSVToText(Task):
    """
    Extract plate in a csv into text/string format.

    Args:
        Task (_type_): _description_
    
    Config Requirements
    path: csv file where it stores the image text file
    Upstream Requirements
    None
    
    Output:
    Dataframe
    id|file_path|text
    
    
    """
    
    def __init__(
        self, path, image_dir, 
        filename_column="image_path",
        text_column = "text",
        split_column = None,
        split = "train",
        name= None
        ):
        super().__init__(name)
        self.path= path
        self.image_dir = image_dir
        self.filename_column = filename_column
        self.text_column = text_column
        self.split_column = split_column
        self.split = split
    
    def run(self, **upstream_outputs):
        source_df = pd.read_csv(self.path)
        rows = []
        for _,row in source_df.iterrows():
            filename = row[self.filename_column]
            
            rows.append({
                "id": os.path.splitext(filename)[0],
                "file_path": os.path.join(self.image_dir, filename),
                "text": row[self.text_column],
                "boxes": None,
                "split": row[self.split_column] if self.split_column else self.split,
            })
        return pd.DataFrame(rows, columns=["id", "file_path", "text", "boxes", "split"])

class LoadYOLOToBoxes(Task):
    """
    This text extract YOLO detection into boxes + class labels, for detection task purposes where the boxes and the label itself are the target.
    
    Config Requirements:
        image_dir
        yaml_file
        labels_dir
        split
        ignore_classes: class names to exclude
    
    Upstream Requirements: None
    
    Output: pandas.DataFrame
        id|file_path |boxes | split
    """
    def __init__(
        self, image_dir, yaml_file, labels_dir, 
        split="train",
        ignore_classes = None, name=None):
        super().__init__(name)
        self.image_dir = image_dir
        self.yaml_file = yaml_file
        self.labels_dir = labels_dir
        self.split = split
        self.ignore_classes = set(ignore_classes or [])
 
    def run(self, **upstream_outputs):
        if not os.path.isdir(self.image_dir):
            return pd.DataFrame(columns=["id", "file_path", "boxes", "split"])
        class_names = _load_class_names(self.yaml_file)
 
        rows = []
        for image_filename in sorted(os.listdir(self.image_dir)):
            stem, ext = os.path.splitext(image_filename)
            if ext.lower() not in (".jpg", ".jpeg", ".png"):
                continue
 
            label_path = os.path.join(self.labels_dir, f"{stem}.txt")
            if not os.path.isfile(label_path):
                continue
 
            boxes = _parse_label_file(label_path, class_names, self.ignore_classes)
            if not boxes:
                continue
 
            rows.append({
                "id": stem,
                "file_path": os.path.join(self.image_dir, image_filename),
                "boxes": boxes,
                "split": self.split,
            })
 
        return pd.DataFrame(rows, columns=["id", "file_path", "boxes", "split"])
    
    
    