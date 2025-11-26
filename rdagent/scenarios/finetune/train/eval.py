import json
from typing import Optional

import yaml

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.finetune.conf import (
    DATA_MAIN_FILE_NAME,
    FT_YAML_FILE_NAME,
    get_clear_ws_cmd,
    get_ft_env,
)
from rdagent.components.coder.finetune.exp import FTTask
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.train.benchmark import run_benchmark


class FTRunnerEvaluator(CoSTEEREvaluator):
    """LLM Fine-tuning specific evaluator that uses LLM Docker environment.

    Supports two task types:
    - "train": Execute LlamaFactory training with train.yaml
    - "data": Execute data processing with main.py
    """

    def evaluate(
        self,
        target_task: FTTask,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: Optional[QueriedKnowledge] = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """Evaluate LLM fine-tuning implementation based on task type."""

        # Determine task type
        task_type = getattr(target_task, "task_type", "train")

        if task_type == "data":
            return self._evaluate_data_task(target_task, implementation)
        else:
            return self._evaluate_train_task(target_task, implementation)

    def _evaluate_train_task(
        self,
        target_task: FTTask,
        implementation: FBWorkspace,
    ) -> CoSTEERSingleFeedback:
        """Evaluate train task: Execute LlamaFactory training."""

        # Use LLM-specific environment with appropriate timeout for training
        timeout_period = getattr(self.scen, "real_full_timeout", lambda: 3600)()

        env = get_ft_env(running_timeout_period=timeout_period)

        # Clean workspace before execution
        implementation.execute(env=env, entry=get_clear_ws_cmd())

        # Check if FT_YAML_FILE_NAME exists
        if FT_YAML_FILE_NAME not in implementation.file_dict:
            return CoSTEERSingleFeedback(
                execution=f"No {FT_YAML_FILE_NAME} found in workspace",
                return_checking="Config file missing",
                code="No valid configuration file",
                final_decision=False,
            )

        # Execute LlamaFactory training
        result = implementation.run(env=env, entry=f"llamafactory-cli train {FT_YAML_FILE_NAME}")
        implementation.running_info.running_time = result.running_time

        # Simple success check: exit code
        training_success = result.exit_code == 0

        # Check for model output files
        workspace_path = implementation.workspace_path
        output_path = workspace_path / "output"
        if not output_path.exists():
            return CoSTEERSingleFeedback(
                execution="Output directory not found",
                return_checking="Output directory not found",
                code="Output directory not found",
                final_decision=False,
            )
        model_output_files = []
        for pattern in ["*.safetensors", "*.bin", "adapter_*"]:
            model_output_files.extend(output_path.glob(pattern))

        # Use open-compass to evaluate the model on benchmark
        benchmark_result = run_benchmark(
            workspace_path=str(workspace_path),
            model_path=output_path,
            model_name=target_task.base_model,
            benchmark_name=target_task.benchmark,
        )

        implementation.running_info.result = benchmark_result

        # Final decision: training succeeded AND model files exist
        final_decision = training_success and len(model_output_files) > 0 and benchmark_result is not None

        # Build minimal feedback
        execution_msg = f"Training {'succeeded' if training_success else 'failed'} (exit_code={result.exit_code})"
        if model_output_files:
            model_msg = f"Found {len(model_output_files)} model output files"
        else:
            model_msg = "No model output files found"

        if benchmark_result:
            model_msg += f"; Benchmark result: {benchmark_result}"

        feedback_msg = f"{execution_msg}. {model_msg}."

        return CoSTEERSingleFeedback(
            execution=feedback_msg,
            return_checking=model_msg,
            code=execution_msg,
            final_decision=final_decision,
        )

    def _evaluate_data_task(
        self,
        target_task: FTTask,
        implementation: FBWorkspace,
    ) -> CoSTEERSingleFeedback:
        """Evaluate data processing task: Execute main.py for full data processing."""

        # Use LLM-specific environment with appropriate timeout for data processing
        timeout_period = getattr(self.scen, "real_full_timeout", lambda: 7200)()

        env = get_ft_env(running_timeout_period=timeout_period)

        # Check if DATA_MAIN_FILE_NAME exists
        if DATA_MAIN_FILE_NAME not in implementation.file_dict:
            return CoSTEERSingleFeedback(
                execution=f"No {DATA_MAIN_FILE_NAME} found in workspace",
                return_checking="Data processing script missing",
                code="No valid data processing script",
                final_decision=False,
            )

        # Build full data processing command
        output_path = "/workspace/sft_dataset.jsonl"
        checkpoint_path = "/workspace/checkpoint.json"
        input_path = "/assets/datasets/seed_data.jsonl"  # Default input path

        data_cmd = (
            f"python {DATA_MAIN_FILE_NAME} "
            f"--input {input_path} "
            f"--output {output_path} "
            f"--checkpoint {checkpoint_path}"
        )

        logger.info(f"Running full data processing: {data_cmd}")
        result = implementation.run(env=env, entry=data_cmd)
        implementation.running_info.running_time = result.running_time

        # Check execution success
        processing_success = result.exit_code == 0
        stdout = result.stdout if result.stdout else ""

        # Check output file
        workspace_path = implementation.workspace_path
        output_file = workspace_path / "sft_dataset.jsonl"
        output_exists = output_file.exists()
        output_valid = False
        output_count = 0

        if output_exists:
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    output_count = len(lines)

                    # Validate alpaca format
                    for line in lines[:10]:  # Check first 10 lines
                        if line.strip():
                            item = json.loads(line)
                            if "instruction" in item and "output" in item:
                                output_valid = True
            except Exception as e:
                logger.warning(f"Error reading output file: {e}")

        # Final decision: processing succeeded AND valid output file exists
        final_decision = processing_success and output_exists and output_valid

        # Build feedback
        execution_msg = f"Data processing {'succeeded' if processing_success else 'failed'} (exit_code={result.exit_code})"
        if output_exists and output_valid:
            output_msg = f"Generated {output_count} SFT samples in alpaca format"
        elif output_exists:
            output_msg = f"Output file exists but format invalid"
        else:
            output_msg = "No output file generated"

        feedback_msg = f"{execution_msg}. {output_msg}."

        implementation.running_info.result = {
            "processing_success": processing_success,
            "output_count": output_count,
            "output_valid": output_valid,
        }

        return CoSTEERSingleFeedback(
            execution=feedback_msg,
            return_checking=output_msg,
            code=execution_msg,
            final_decision=final_decision,
        )
