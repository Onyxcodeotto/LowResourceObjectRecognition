"""Silver layer: a config-driven transform DAG executed by Prefect.

Task classes live in ``src/transforms/tasks/`` and stay framework-free. This
module is the only place Prefect is imported.

Config shape::

    {
      "datasets": [
        {
          "name": "domain",
          "chain": [
            {"id": "load",   "type": "YOLO2Boxes",     "params": {...}},
            {"id": "sanity", "type": "BboxSanityCheck", "params": {...},
             "depends_on": ["load"]},
            ...
          ]
        }
      ],
      "merge":  {"type": "Merge",              "params": {...}},
      "output": {"type": "GenerateOCRManifest", "params": {...}}
    }

Each chain node may also carry ``"prefect": {...}`` - options passed straight to
``@task`` (retries, retry_delay_seconds, cache_policy, timeout_seconds, ...).
"""
import json

from prefect import flow, get_run_logger, task

from .pipe import Pipe
from ..transforms.tasks.task_registry import TASK_REGISTRY


def _prefect_task(task_obj, **task_opts):
    """Wrap a Task instance's ``run`` as a Prefect task."""

    @task(name=task_obj.name, **task_opts)
    def _run(**upstream_outputs):
        return task_obj.run(**upstream_outputs)

    return _run


@flow(name="silver-recognition")
def silver_recognition_flow(config: dict):
    log = get_run_logger()
    dataset_ends = {}

    for dataset in config["datasets"]:
        results = {}
        for node in dataset["chain"]:
            instance = TASK_REGISTRY[node["type"]](**node.get("params", {}))
            runner = _prefect_task(instance, **node.get("prefect", {}))
            deps = {dep: results[dep] for dep in node.get("depends_on", [])}
            results[node["id"]] = runner(**deps) if deps else runner()
        dataset_ends[dataset["name"]] = results[dataset["chain"][-1]["id"]]
        log.info("dataset %s: assembled %d nodes", dataset["name"], len(dataset["chain"]))

    merge_cfg = config["merge"]
    merge_instance = TASK_REGISTRY[merge_cfg["type"]](**merge_cfg.get("params", {}))
    merged = _prefect_task(merge_instance)(**dataset_ends)

    out_cfg = config["output"]
    out_instance = TASK_REGISTRY[out_cfg["type"]](**out_cfg.get("params", {}))
    return _prefect_task(out_instance)(unified=merged)


class SilverPipeline(Pipe):
    """Runs the silver transform DAG from a config dict (see module docstring)."""

    def run(self, config):
        return silver_recognition_flow(config)


def run_from_file(config_path):
    with open(config_path) as handle:
        return SilverPipeline().run(json.load(handle))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.pipelines.silver_pipeline <config.json>")
    run_from_file(sys.argv[1])
