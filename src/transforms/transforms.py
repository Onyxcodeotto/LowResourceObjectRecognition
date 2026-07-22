class CustomDAG:
    """
    Assemble task instances into a pipeline of tasks. Mimics DAG from Airflow.
    """
    
    def __init__(self, name):
        self.name = name
        self.tasks = {}
        self.dependencies = {}
        
    def add_task(self, task, dependencies=None):
        if task.name in self.tasks:
            raise ValueError(f"Task with name '{task.name}' already exists in the transform.")
        self.tasks[task.name] = task
        self.dependencies[task.name] = list(dependencies) if dependencies else []
        return self
    
    def run(self, output_task = None):
        order = self._generate_topological_order()
       
        results = {}
        for task_name in order:
           task = self.tasks[task_name]
           upstream_outputs = {dep: results[dep] for dep in self.dependencies[task_name]}
           results[task_name] = task.run(**upstream_outputs)
           
           
        if output_task:
            return results[output_task]

        # Default: the DAG's single terminal task (no downstream dependents).
        all_deps = {dep for deps in self.dependencies.values() for dep in deps}
        terminal_tasks = [t for t in self.tasks if t not in all_deps]
        if len(terminal_tasks) != 1:
            raise ValueError(
                f"DAG '{self.name}' has {len(terminal_tasks)} terminal tasks "
                f"({terminal_tasks}); pass output_task explicitly."
            )
        return results[terminal_tasks[0]]
        

    def _generate_topological_order(self):
        # Implementation for generating the execution order of tasks, using depth first search recursion algorthim.
        visited, marked, order = set(), set(), []
        def visit(node):
            if node in visited:
                return
            if node in marked:
                raise ValueError(f"Cyclic dependency detected at task '{node}'")
            marked.add(node)
            for dep in self.dependencies.get(node, []):
                if dep not in self.tasks:
                    raise ValueError(f"Dependency '{dep}' for task '{node}' not found in the transform.")
                visit(dep)
            marked.remove(node)
            visited.add(node)
            order.append(node)
        for task in self.tasks:
            visit(task)
        return order
                
        