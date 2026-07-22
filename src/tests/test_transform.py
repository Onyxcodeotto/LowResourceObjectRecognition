

import unittest

import pandas as pd

from ..transforms.task import Task
from ..transforms.transforms import CustomDAG

### Custom task to test DAG and task
class _Sum(Task):
    def run(self, **upstream_outputs):
        return sum(upstream_outputs.values())

class _AddOne(Task):
    def run(self, **upstream_outputs):
        return self.single_input(upstream_outputs) + 1
    
    
class TestDag(unittest.TestCase):
    def test_join_of_two_roots(self):
        class SourceA(Task):
            def run(self, **upstream_outputs):
                return 10
 
        class SourceB(Task):
            def run(self, **upstream_outputs):
                return 5
 
        dag = CustomDAG(name="join_test")
        dag.add_task(SourceA(name="a"))
        dag.add_task(SourceB(name="b"))
        dag.add_task(_Sum(name="sum"), dependencies=["a", "b"])
 
        result = dag.run(output_task="sum")
        self.assertEqual(result, 15)
 
    def test_cycle_detection(self):
        dag = CustomDAG(name="cyclic")
        dag.add_task(_AddOne(name="x"), dependencies=["y"])
        dag.add_task(_AddOne(name="y"), dependencies=["x"])
        with self.assertRaises(ValueError):
            dag.run(output_task="x")