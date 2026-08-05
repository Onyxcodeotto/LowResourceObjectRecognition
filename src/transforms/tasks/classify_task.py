

from task import Task

class ClassifyTask(Task):
    def __init__(self, model):
        self.model = model
