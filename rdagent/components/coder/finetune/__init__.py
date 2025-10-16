"""
LLM Fine-tuning CoSTEER Implementation

This module provides fine-tuning specific components for the CoSTEER framework,
including evaluators and evolving strategies.
"""

import re
from pathlib import Path

import yaml

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiEvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.components.coder.finetune.conf import FTCoderCoSTEERSettings
from rdagent.components.coder.finetune.eval import LLMFinetuneEvaluator
from rdagent.components.coder.finetune.exp import TrainingTask
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent


class LLMFinetuneEvolvingStrategy(MultiProcessEvolvingStrategy):
    """LLM Fine-tuning specific evolving strategy"""

    def __init__(self, scen: Scenario, settings, *args, **kwargs):
        super().__init__(scen, settings)

        # Lazy import to avoid circular dependency
        from rdagent.scenarios.finetune.scen.llama_factory_manager import (
            get_llama_factory_manager,
        )

        self.llama_factory_manager = get_llama_factory_manager()

    def implement_one_task(
        self,
        target_task: Task,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        """Convert JSON hypothesis to LlamaFactory YAML config"""

        # Get hypothesis from task (attached by FTHypothesis2Experiment)
        hypothesis = getattr(target_task, "_parent_experiment_hypothesis", None)
        
        if hypothesis and hasattr(hypothesis, "hypothesis_json") and hypothesis.hypothesis_json:
            # New path: Direct JSON to YAML conversion (no LLM needed)
            logger.info("Converting JSON hypothesis to YAML configuration")
            config_yaml = self._convert_json_to_yaml(
                hypothesis_json=hypothesis.hypothesis_json,
                base_model=getattr(target_task, "base_model"),
                dataset=getattr(target_task, "dataset"),
                debug_mode=True,
            )
        else:
            # Fallback: Use legacy method (backward compatibility)
            logger.warning("No hypothesis JSON found, falling back to legacy config generation")
            task_info = target_task.get_task_information()
            similar_knowledge = (
                queried_knowledge.task_to_similar_task_successful_knowledge.get(task_info, []) if queried_knowledge else []
            )
            failed_knowledge = (
                queried_knowledge.task_to_former_failed_traces.get(task_info, ([], None))
                if queried_knowledge
                else ([], None)
            )
            
            config_yaml = self._generate_llamafactory_config_with_llm(
                base_model=getattr(target_task, "base_model"),
                finetune_method=getattr(target_task, "finetune_method"),
                dataset=getattr(target_task, "dataset"),
                debug_mode=True,
                task_info=task_info,
                similar_knowledge=similar_knowledge,
                failed_knowledge=failed_knowledge[0],
                prev_feedback=prev_task_feedback,
                workspace=workspace,
            )

        return {"train.yaml": config_yaml}
    
    def _convert_json_to_yaml(
        self,
        hypothesis_json: dict,
        base_model: str,
        dataset: str,
        debug_mode: bool = True,
    ) -> str:
        """Convert JSON hypothesis to LlamaFactory YAML configuration
        
        Direct conversion without LLM - simply flattens JSON and adds system parameters.
        
        Args:
            hypothesis_json: JSON hypothesis from ExpGen
            base_model: Base model name
            dataset: Dataset name
            debug_mode: Whether to use debug settings
            
        Returns:
            YAML configuration string
        """
        # Flatten nested JSON to config dict
        config = self._flatten_dict(hypothesis_json)
        
        # Remove non-parameter fields (like reasoning, explanations, etc.)
        meta_keys = ["reasoning", "reason", "rationale", "why", "explanation", 
                     "expected_outcome", "outcome", "changes", "changes_from_previous"]
        for key in meta_keys:
            config.pop(key, None)
        
        # Add system-required parameters
        config.update({
            "model_name_or_path": f"/assets/models/{base_model}",
            "dataset": dataset,
            "dataset_dir": "/assets/datasets/",
            "output_dir": "/workspace/output",
            "do_train": True,
            "overwrite_output_dir": True,
            "logging_steps": 10,
            "save_steps": 500,
            "plot_loss": True,
            "report_to": "none",
        })
        
        # Debug mode settings
        if debug_mode:
            config["max_samples"] = 100
            if "num_train_epochs" in config:
                config["num_train_epochs"] = min(config["num_train_epochs"], 1)
        
        # Hardware adaptation
        config.setdefault("bf16", True)
        
        # Generate YAML
        yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
        logger.info(f"Generated YAML config with {len(config)} parameters")
        return yaml_str
    
    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = "_") -> dict:
        """Recursively flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                # Recursively flatten
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _generate_llamafactory_config_with_llm(
        self,
        base_model: str,
        finetune_method: str,
        dataset: str,
        debug_mode: bool = True,
        task_info: str = "",
        similar_knowledge: list = None,
        failed_knowledge: list = None,
        prev_feedback=None,
        workspace=None,
    ) -> str:
        """Generate LlamaFactory configuration YAML using LLM"""

        # Prepare knowledge context
        similar_knowledge_str = ""
        if similar_knowledge:
            similar_knowledge_str = "\n".join(
                [
                    f"### Similar Implementation {i+1}:\n{knowledge.target_task.get_task_information()}\n```yaml\n{knowledge.implementation.file_dict.get('train.yaml', '')}\n```"
                    for i, knowledge in enumerate(similar_knowledge)
                ]
            )

        failed_knowledge_str = ""
        if failed_knowledge and isinstance(failed_knowledge, (list, tuple)) and len(failed_knowledge) > 0:
            # Handle both list of knowledge and tuple format (knowledge_list, None)
            knowledge_list = failed_knowledge[0] if isinstance(failed_knowledge, tuple) else failed_knowledge
            if knowledge_list:
                failed_knowledge_str = "\n".join(
                    [
                        f"### Failed Attempt {i+1}:\n```yaml\n{knowledge.implementation.file_dict.get('train.yaml', '')}\n```\n**Feedback:** {knowledge.feedback}"
                        for i, knowledge in enumerate(knowledge_list)
                    ]
                )

        # Query LLaMA Factory parameters for the specific method
        method_params_desc = self.llama_factory_manager.format_method_params(finetune_method)

        # Use fixed Docker paths for simplicity
        models_path = "/assets/models/"
        datasets_path = "/assets/datasets/"

        # Generate prompts using templates with all required parameters
        # TODO: give exp_gen(natural language) here
        system_prompt = T("components.coder.finetune.prompts:finetune_coder.system").r(
            task_desc=task_info,
            finetune_method=finetune_method,
            similar_knowledge=similar_knowledge if similar_knowledge else [],
            failed_knowledge=failed_knowledge[0] if isinstance(failed_knowledge, tuple) and failed_knowledge[0] else [],
            method_params=method_params_desc,
        )

        user_prompt = T("components.coder.finetune.prompts:finetune_coder.user").r(
            latest_code=(workspace.file_dict.get("train.yaml", "") if workspace and prev_feedback else ""),
            latest_feedback=str(prev_feedback) if prev_feedback else "",
            finetune_method=finetune_method,
            base_model=base_model,
            dataset_name=dataset,
            models_path=models_path,
            datasets_path=datasets_path,
        )

        # Call LLM to generate config
        try:
            api = APIBackend()
            response = api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=False,
            )

            # Extract YAML content from response
            # Try markdown code block first (standard format from improved prompt)
            match = re.search(r"```(?:yaml)?\s*\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_yaml = match.group(1).strip()
                try:
                    yaml.safe_load(extracted_yaml)
                    logger.info("Extracted YAML from markdown code block")
                    return extracted_yaml
                except yaml.YAMLError as e:
                    logger.warning(f"Extracted YAML is invalid: {e}")
                    raise RuntimeError(f"Invalid YAML in code block: {e}")

            # Fallback: try to use entire response as YAML
            try:
                yaml.safe_load(response)
                logger.info("Using entire response as YAML")
                return response.strip()
            except yaml.YAMLError as e:
                logger.error(f"Failed to parse response as YAML: {e}")
                raise RuntimeError(f"Failed to extract valid YAML from LLM response: {e}")

        except Exception as e:
            logger.error(f"Failed to generate config with LLM: {e}")
            raise RuntimeError(f"LLM config generation failed: {e}")

    def assign_code_list_to_evo(self, code_list: list[dict[str, str]], evo):
        """Assign generated code to the evolving experiment"""
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = evo.experiment_workspace
            evo.sub_workspace_list[index].inject_files(**code_list[index])
        return evo


class LLMFinetuneCoSTEER(CoSTEER):
    """LLM Fine-tuning CoSTEER implementation"""

    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        settings = FTCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator(LLMFinetuneEvaluator(scen=scen), scen=scen)
        es = LLMFinetuneEvolvingStrategy(scen=scen, settings=settings)

        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            with_knowledge=False,
            knowledge_self_gen=False,
            max_loop=FT_RD_SETTING.coder_max_loop if hasattr(FT_RD_SETTING, "coder_max_loop") else 5,
            **kwargs,
        )
