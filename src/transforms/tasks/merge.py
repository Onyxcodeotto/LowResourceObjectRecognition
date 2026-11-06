"""Fan-in task: union per-dataset frames into one branch dataset."""
import logging

import pandas as pd

from src.transforms.tasks.task import Task

logger = logging.getLogger(__name__)

# Grand schema for the recognition (OCR) branch. Detection passes its own list.
OCR_SCHEMA = [
    "image_path", "text", "source_type", "source_identifier",
    "split", "width", "height",
]


class Merge(Task):
    """Union one or more per-dataset frames into a single branch dataset.

    Args:
        rename:   dict applied to every frame's columns before the union
                  (e.g. ``{"file_path": "image_path"}``).
        schema:   optional list of columns to guarantee exist (filled with
                  ``pd.NA``) and to place first in the output. Extra columns are
                  kept, appended after the schema columns. ``None`` = keep every
                  frame's columns as-is.
        dedup_on: column (or list) to drop duplicate rows on. ``None`` = no dedup.

    Input:  ``**upstream_outputs`` - any number of DataFrames, keyed by name.
    Output: one DataFrame.
    """

    def __init__(self, rename=None, schema=None, dedup_on=None, name=None):
        super().__init__(name)
        self.rename = rename or {}
        self.schema = list(schema) if schema else None
        self.dedup_on = dedup_on

    def run(self, **upstream_outputs):
        if not upstream_outputs:
            raise ValueError("Merge: needs at least one upstream frame")

        frames = []
        for source_name, frame in upstream_outputs.items():
            frame = frame.rename(columns=self.rename)
            if self.schema:
                for column in self.schema:
                    if column not in frame.columns:
                        frame[column] = pd.NA
            logger.info("Merge: %s contributes %d rows", source_name, len(frame))
            frames.append(frame)

        merged = pd.concat(frames, ignore_index=True)

        if self.dedup_on is not None:
            before = len(merged)
            merged = merged.drop_duplicates(subset=self.dedup_on, ignore_index=True)
            logger.info("Merge: %d -> %d rows after dedup on %s",
                        before, len(merged), self.dedup_on)
        else:
            logger.info("Merge: %d rows", len(merged))

        if self.schema:
            ordered = self.schema + [c for c in merged.columns if c not in self.schema]
            merged = merged[ordered]
        return merged
