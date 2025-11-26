"""
LLM Fine-tuning Experiment Components

Defines tasks for LLM fine-tuning following data science pattern.
Supports two task types:
- "train": Generate train.yaml for LlamaFactory training
- "data": Generate main.py for COT-Self-Instruct data processing pipeline
"""

from typing import Literal

from rdagent.components.coder.CoSTEER.task import CoSTEERTask


# Because we use isinstance to distinguish between different types of tasks, we need to use sub classes to represent different types of tasks
class FTTask(CoSTEERTask):
    """Task class for LLM fine-tuning operations - follows data science pattern

    Supports two task types:
    - "train": Traditional training task, generates train.yaml
    - "data": Data processing task, generates main.py for COT-Self-Instruct pipeline
    """

    def __init__(
        self,
        base_model: str,
        description: str,
        benchmark: str,
        task_type: Literal["train", "data"] = "train",
        *args,
        **kwargs,
    ) -> None:
        task_name = "LLM-Fine-Tuning" if task_type == "train" else "LLM-Data-Processing"
        super().__init__(name=task_name, description=description, *args, **kwargs)
        self.base_model = base_model
        self.benchmark = benchmark
        self.task_type = task_type

    def get_task_information(self) -> str:
        """Get task information for coder prompt generation"""
        return f"""name: {self.name}
description: {self.description}
base_model: {self.base_model}
task_type: {self.task_type}
"""
