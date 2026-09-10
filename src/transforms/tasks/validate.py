"""Validation tasks for the silver pipeline.
    This file contains all validation needed to make sure data isnt invalid or broken.
"""
import logging

from src.transforms.tasks.task import Task

logger = logging.getLogger(__name__)


class BboxSanityCheck(Task):
    """Flag rows whose YOLO boxes are geometrically invalid.

    Does not drop rows - adds ``valid`` (bool) and ``invalid_reason`` (str, ""
    when valid) so downstream tasks decide what to do with bad labels.

    Input:  DataFrame with a ``boxes`` column, each cell a list of
            ``(cx, cy, w, h, class_name)`` in normalised YOLO coords.
    Output: same rows + ``valid``, ``invalid_reason``.
    """

    def __init__(self, coord_bounds=(0.0, 1.0), min_box_area=0.0,
                 require_at_least_one=True, name=None):
        super().__init__(name)
        self.lo, self.hi = coord_bounds
        self.min_box_area = min_box_area
        self.require_at_least_one = require_at_least_one

    def _check_box(self, box):
        try:
            cx, cy, w, h = (float(v) for v in tuple(box)[:4])
        except (TypeError, ValueError):
            return "unparseable_box"
        if not (self.lo <= cx <= self.hi and self.lo <= cy <= self.hi):
            return "center_out_of_bounds"
        if w <= 0 or h <= 0:
            return "non_positive_size"
        if (cx - w / 2 < self.lo or cx + w / 2 > self.hi
                or cy - h / 2 < self.lo or cy + h / 2 > self.hi):
            return "box_exceeds_image"
        if w * h < self.min_box_area:
            return "below_min_area"
        return ""

    def _check_row(self, boxes):
        if not isinstance(boxes, (list, tuple)):
            boxes = []
        if self.require_at_least_one and len(boxes) == 0:
            return "no_boxes"
        for box in boxes:
            reason = self._check_box(box)
            if reason:
                return reason
        return ""

    def run(self, **upstream_outputs):
        df = self.single_input(upstream_outputs).copy()
        df["invalid_reason"] = [self._check_row(b) for b in df["boxes"]]
        df["valid"] = df["invalid_reason"] == ""

        n_bad = int((~df["valid"]).sum())
        if n_bad:
            logger.warning(
                "BboxSanityCheck: %d/%d rows invalid: %s",
                n_bad, len(df),
                df.loc[~df["valid"], "invalid_reason"].value_counts().to_dict(),
            )
        return df
