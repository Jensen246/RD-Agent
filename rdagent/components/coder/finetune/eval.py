"""
LLM Fine-tuning Evaluation Components

Provides simplified evaluation:
- For train tasks: parameter filtering + micro-batch testing
- For data tasks: debug mode execution + output validation
"""

import json
from pathlib import Path

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.finetune.conf import (
    DATA_MAIN_FILE_NAME,
    FT_YAML_FILE_NAME,
    get_ft_env,
)
from rdagent.components.coder.finetune.unified_validator import create_unified_validator
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

DIRNAME = Path(__file__).absolute().resolve().parent


class FTCoderEvaluator(CoSTEEREvaluator):
    """Evaluator for LLM fine-tuning implementations with simplified validation

    Supports two task types:
    - "train": Validates train.yaml with parameter filtering + micro-batch test
    - "data": Validates main.py with debug mode execution + output validation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_validator = create_unified_validator()

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """Evaluate LLM fine-tuning implementation based on task type"""

        task_info = target_task.get_task_information()

        # Check task history
        if queried_knowledge is not None:
            if task_info in queried_knowledge.success_task_to_knowledge_dict:
                return queried_knowledge.success_task_to_knowledge_dict[task_info].feedback
            elif task_info in queried_knowledge.failed_task_info_set:
                return CoSTEERSingleFeedback(
                    execution="Task failed too many times, skipping.",
                    return_checking="Task failed too many times, skipping.",
                    code="Task failed too many times, skipping.",
                    final_decision=False,
                )

        # Determine task type
        task_type = getattr(target_task, "task_type", "train")

        if task_type == "data":
            return self._evaluate_data_task(target_task, implementation, queried_knowledge)
        else:
            return self._evaluate_train_task(target_task, implementation, queried_knowledge)

    def _evaluate_train_task(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
    ) -> CoSTEERSingleFeedback:
        """Evaluate train task: parameter filtering + micro-batch test"""

        env = get_ft_env(
            running_timeout_period=self.scen.real_debug_timeout() if hasattr(self.scen, "real_debug_timeout") else 3600,
        )
        config_yaml = implementation.file_dict.get(FT_YAML_FILE_NAME, "")
        if not config_yaml:
            return CoSTEERSingleFeedback(
                execution=f"No {FT_YAML_FILE_NAME} found",
                return_checking="Configuration file missing",
                code="No valid configuration file",
                final_decision=False,
            )

        # Two-step validation: parameter filtering + micro-batch test
        validation_result = self.config_validator.validate_and_test(
            config_yaml=config_yaml, workspace=implementation, env=env
        )

        # Update config with filtered version
        if validation_result.filtered_config != config_yaml:
            implementation.inject_files(**{FT_YAML_FILE_NAME: validation_result.filtered_config})

        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[target_task.get_task_information()]
            if queried_knowledge is not None
            else []
        )

        system_prompt = T(".prompts:finetune_eval.system").r(
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
        )
        user_prompt = T(".prompts:finetune_eval.user").r(
            scenario=self.scen.get_scenario_all_desc(),
            task_desc=target_task.get_task_information(),
            stdout=validation_result.execution_output or "No output",
            code_yaml=implementation.file_dict[FT_YAML_FILE_NAME],
            workspace_files="\n".join(
                [
                    f"- {file.name} ({file.stat().st_size} bytes)"
                    for file in implementation.workspace_path.rglob("*")
                    if file.is_file() and "checkpoint" not in file.absolute().as_posix()
                ]
            ),
        )
        return build_cls_from_json_with_retry(
            CoSTEERSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=CoSTEERSingleFeedback.val_and_update_init_dict,
        )

    def _evaluate_data_task(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
    ) -> CoSTEERSingleFeedback:
        """Evaluate data processing task: debug mode execution + output validation"""

        main_py = implementation.file_dict.get(DATA_MAIN_FILE_NAME, "")
        if not main_py:
            return CoSTEERSingleFeedback(
                execution=f"No {DATA_MAIN_FILE_NAME} found",
                return_checking="Data processing script missing",
                code="No valid data processing script",
                final_decision=False,
            )

        env = get_ft_env(
            running_timeout_period=self.scen.real_debug_timeout() if hasattr(self.scen, "real_debug_timeout") else 3600,
        )

        # Run main.py in debug mode with limited samples
        debug_output_path = "/workspace/debug_output.jsonl"
        debug_checkpoint_path = "/workspace/debug_checkpoint.json"
        input_path = "/assets/datasets/seed_data.jsonl"  # Default input path

        # Build debug command
        debug_cmd = (
            f"python {DATA_MAIN_FILE_NAME} "
            f"--input {input_path} "
            f"--output {debug_output_path} "
            f"--checkpoint {debug_checkpoint_path} "
            f"--debug --max_samples 5"
        )

        logger.info(f"Running data processing debug: {debug_cmd}")
        execution_result = implementation.run(env=env, entry=debug_cmd)

        # Collect execution output
        stdout = execution_result.stdout if execution_result.stdout else ""
        exit_code = execution_result.exit_code

        # Check output file
        output_files_info = []
        output_valid = False
        output_content_preview = ""

        # List workspace files after execution
        try:
            for file in implementation.workspace_path.rglob("*"):
                if file.is_file() and "checkpoint" not in file.absolute().as_posix():
                    output_files_info.append(f"- {file.name} ({file.stat().st_size} bytes)")

                    # Check output file format
                    if file.name == "debug_output.jsonl" or file.name.endswith("_output.jsonl"):
                        try:
                            with open(file, "r", encoding="utf-8") as f:
                                lines = f.readlines()[:3]  # Preview first 3 lines
                                output_content_preview = "".join(lines)

                                # Validate alpaca format
                                for line in lines:
                                    if line.strip():
                                        item = json.loads(line)
                                        if "instruction" in item and "output" in item:
                                            output_valid = True
                        except Exception as e:
                            output_content_preview = f"Error reading output: {e}"
        except Exception as e:
            logger.warning(f"Error listing workspace files: {e}")

        # Get similar successful knowledge for prompt
        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[target_task.get_task_information()]
            if queried_knowledge is not None
            else []
        )

        # Use data evaluation prompts
        system_prompt = T("rdagent.components.coder.finetune.data_prompts:data_eval.system").r(
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
        )
        user_prompt = T("rdagent.components.coder.finetune.data_prompts:data_eval.user").r(
            scenario=self.scen.get_scenario_all_desc(),
            task_desc=target_task.get_task_information(),
            code_python=main_py,
            stdout=stdout,
            output_files="\n".join(output_files_info) if output_files_info else "No output files generated",
        )

        return build_cls_from_json_with_retry(
            CoSTEERSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=CoSTEERSingleFeedback.val_and_update_init_dict,
        )
