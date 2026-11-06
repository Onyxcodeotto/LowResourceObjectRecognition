"""Terminal sink tasks: write a branch dataset to its training-ready layout."""
import logging
import os
import shutil

import yaml

from src.transforms.tasks.task import Task

logger = logging.getLogger(__name__)

_IMAGE_MODES = ("copy", "symlink", "none")


class GenerateYOLO(Task):
    """Write the detection-branch dataset in YOLO training layout.

    Produces under ``out_dir``::

        <split>/images/<stem>.<ext>   (copied / symlinked / skipped per image_mode)
        <split>/labels/<stem>.txt     (one line per box: "cls_id cx cy w h")
        data.yaml                     (path, train/val/test, nc, names)

    ``boxes`` cells are lists of ``(cx, cy, w, h, class_name)`` in normalised
    YOLO coords. Returns the input frame unchanged.

    Args:
        out_dir:      output root.
        class_names:  ordered list; index = YOLO class id. If omitted and
                      ``class_names_yaml`` is not given, it is derived from the
                      data (sorted unique names) with a warning.
        class_names_yaml: path to a YAML with a ``names`` key to load the list from.
        image_mode:   "copy" (default), "symlink", or "none" (labels only).
        image_col / boxes_col / split_col: column names.
        default_split: split used when the split value is missing / NA.
    """

    def __init__(self, out_dir, class_names=None, class_names_yaml=None,
                 image_mode="copy", image_col="image_path", boxes_col="boxes",
                 split_col="split", default_split="train", name=None):
        super().__init__(name)
        if image_mode not in _IMAGE_MODES:
            raise ValueError(f"image_mode must be one of {_IMAGE_MODES}")
        self.out_dir = out_dir
        self.image_mode = image_mode
        self.image_col = image_col
        self.boxes_col = boxes_col
        self.split_col = split_col
        self.default_split = default_split

        if class_names is None and class_names_yaml:
            with open(class_names_yaml) as handle:
                class_names = yaml.safe_load(handle)["names"]
        self.class_names = list(class_names) if class_names else None

    def _resolve_class_names(self, df):
        if self.class_names:
            return self.class_names
        seen = set()
        for boxes in df[self.boxes_col]:
            for box in boxes or []:
                if len(box) > 4:
                    seen.add(box[4])
        derived = sorted(seen)
        logger.warning(
            "GenerateYOLO: no class_names given, derived %d from data: %s "
            "(pass an explicit list for a stable class order)",
            len(derived), derived,
        )
        return derived

    def _place_image(self, src, dst):
        if self.image_mode == "none":
            return True
        if not os.path.isfile(src):
            logger.warning("GenerateYOLO: image missing, skipping row: %s", src)
            return False
        if os.path.exists(dst):
            return True
        if self.image_mode == "copy":
            shutil.copy2(src, dst)
        else:  # symlink
            os.symlink(os.path.abspath(src), dst)
        return True

    def run(self, **upstream_outputs):
        df = self.single_input(upstream_outputs)
        class_names = self._resolve_class_names(df)
        class_to_id = {name: idx for idx, name in enumerate(class_names)}

        splits_seen = set()
        rows_written = 0
        boxes_dropped = 0

        for record in df.itertuples(index=False):
            src = getattr(record, self.image_col)
            split = getattr(record, self.split_col, None)
            if split is None or (isinstance(split, float) and split != split):
                split = self.default_split
            split = str(split)

            img_dir = os.path.join(self.out_dir, split, "images")
            lbl_dir = os.path.join(self.out_dir, split, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            stem, ext = os.path.splitext(os.path.basename(src))
            if not self._place_image(src, os.path.join(img_dir, stem + ext)):
                continue

            lines = []
            for box in getattr(record, self.boxes_col) or []:
                cx, cy, w, h = (float(v) for v in tuple(box)[:4])
                cls_name = box[4] if len(box) > 4 else None
                if cls_name not in class_to_id:
                    boxes_dropped += 1
                    continue
                lines.append(f"{class_to_id[cls_name]} {cx} {cy} {w} {h}")
            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as handle:
                handle.write("\n".join(lines))

            splits_seen.add(split)
            rows_written += 1

        data_yaml = {
            "path": os.path.abspath(self.out_dir),
            "nc": len(class_names),
            "names": class_names,
        }
        for split in sorted(splits_seen):
            data_yaml[split] = os.path.join(split, "images")
        with open(os.path.join(self.out_dir, "data.yaml"), "w") as handle:
            yaml.safe_dump(data_yaml, handle, sort_keys=False)

        logger.info(
            "GenerateYOLO: wrote %d rows across splits %s (%d boxes dropped, unknown class) -> %s",
            rows_written, sorted(splits_seen), boxes_dropped, self.out_dir,
        )
        return df
