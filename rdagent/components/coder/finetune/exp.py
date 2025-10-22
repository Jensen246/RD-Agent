"""
LLM Fine-tuning Experiment Components

Defines tasks for LLM fine-tuning following data science pattern.
"""

from rdagent.components.coder.CoSTEER.task import CoSTEERTask


# Because we use isinstance to distinguish between different types of tasks, we need to use sub classes to represent different types of tasks
class TrainingTask(CoSTEERTask):
    """Training task class for LLM fine-tuning operations - follows data science pattern"""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        finetune_method: str = "lora",
        dataset: str = "default",
        name: str = "Training",
        description: str = "",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name=name, description=description, *args, **kwargs)
        self.base_model = base_model
        self.finetune_method = finetune_method
        self.dataset = dataset

    def get_task_information(self) -> str:
        """Get task information for coder prompt generation"""
        task_desc = f"""name: {self.name}
description: {self.description}
base_model: {self.base_model}
finetune_method: {self.finetune_method}
dataset: {self.dataset}
"""
        return task_desc
