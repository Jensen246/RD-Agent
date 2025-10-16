"""LLM Fine-tuning Base Classes"""

import json
from typing import Any, Dict, List, Literal, Optional

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.components.coder.finetune.exp import TrainingTask
from rdagent.core.proposal import ExpGen, Hypothesis, Hypothesis2Experiment, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.finetune.experiment.experiment import FTExperiment
from rdagent.scenarios.finetune.scen.llama_factory_manager import (
    get_llama_factory_manager,
)
from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen
from rdagent.scenarios.finetune.scen.utils import extract_dataset_info
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T

COMPONENT = Literal["Training"]


class FTHypothesis(Hypothesis):
    """LLM fine-tuning hypothesis - free-form JSON format."""

    def __init__(
        self,
        hypothesis_json: Optional[Dict[str, Any]] = None,
        # Backward compatibility fields
        base_model: Optional[str] = None,
        finetune_method: Optional[str] = None,
        quantization: str = "none",
        hypothesis: str | None = None,
        reason: str | None = None,
        concise_reason: str | None = None,
        concise_observation: str | None = None,
        concise_justification: str | None = None,
        concise_knowledge: str | None = None,
    ) -> None:
        super().__init__(
            hypothesis, reason, concise_reason, concise_observation, concise_justification, concise_knowledge
        )
        self.hypothesis_json = hypothesis_json or {}
        
        # Extract for backward compatibility
        self.base_model = base_model or FT_RD_SETTING.base_model
        self.finetune_method = finetune_method or self._find_finetune_type()
        self.quantization = quantization or self._find_quantization()
        
        # Auto-generate hypothesis string
        if hypothesis is None:
            reasoning = self._find_reasoning()
            if reasoning:
                self.hypothesis = reasoning[:200]
            else:
                method_desc = self.finetune_method
                if self.quantization != "none":
                    method_desc += f" with {self.quantization} quantization"
                self.hypothesis = f"Fine-tune {self.base_model} using {method_desc}"
        
        # Extract reason
        if reason is None:
            self.reason = self._find_reasoning()
    
    def _find_in_dict(self, d: Any, keys: List[str]) -> Any:
        """Recursively search for any of the keys in nested dict"""
        if not isinstance(d, dict):
            return None
        
        # Check direct keys
        for key in keys:
            if key in d:
                return d[key]
        
        # Recursively search nested dicts
        for value in d.values():
            if isinstance(value, dict):
                result = self._find_in_dict(value, keys)
                if result is not None:
                    return result
        
        return None
    
    def _find_finetune_type(self) -> str:
        """Find finetuning type in JSON (flexible key names)"""
        result = self._find_in_dict(
            self.hypothesis_json,
            ["finetuning_type", "method", "finetune_method", "ft_type", "fine_tuning_method"]
        )
        return result if result in ["lora", "freeze", "full"] else "lora"
    
    def _find_quantization(self) -> str:
        """Find quantization info in JSON"""
        quant_bit = self._find_in_dict(
            self.hypothesis_json,
            ["quantization_bit", "quant_bit", "quantization", "quant"]
        )
        if quant_bit == 4 or quant_bit == "4" or quant_bit == "4bit":
            return "4bit"
        elif quant_bit == 8 or quant_bit == "8" or quant_bit == "8bit":
            return "8bit"
        return "none"
    
    def _find_reasoning(self) -> str:
        """Find reasoning in JSON"""
        result = self._find_in_dict(
            self.hypothesis_json,
            ["reasoning", "reason", "rationale", "why", "explanation"]
        )
        return str(result) if result else ""

    def __str__(self) -> str:
        return json.dumps(self.hypothesis_json, indent=2) if self.hypothesis_json else f"Fine-tuning {self.base_model}"


class FTHypothesis2Experiment(Hypothesis2Experiment):
    """Convert LLM fine-tuning hypothesis to experiment."""

    def convert(self, hypothesis: FTHypothesis, trace: Trace) -> FTExperiment:
        """Convert hypothesis to executable experiment."""
        logger.info(f"Converting hypothesis: {hypothesis.base_model} with {hypothesis.finetune_method}")

        # Ensure the selected model is downloaded if it wasn't specified initially
        from rdagent.scenarios.finetune.utils import ensure_ft_assets_exist

        ensure_ft_assets_exist(model=hypothesis.base_model, check_model=True)

        # Combine method and quantization for task description
        method_desc = hypothesis.finetune_method
        if hypothesis.quantization != "none":
            method_desc += f" with {hypothesis.quantization} quantization"

        task = TrainingTask(
            base_model=hypothesis.base_model,
            finetune_method=hypothesis.finetune_method,
            dataset=FT_RD_SETTING.dataset,
            name="Training",
            description=f"Fine-tune {hypothesis.base_model} using {method_desc}",
        )
        
        # Attach hypothesis to task for Coder to access decisions
        task._parent_experiment_hypothesis = hypothesis

        return FTExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)


class LLMFinetuneExpGen(ExpGen):
    """LLM fine-tuning experiment generator with full parameter control."""

    def __init__(self, scen: LLMFinetuneScen):
        super().__init__(scen)
        self.llama_manager = get_llama_factory_manager()

    def gen(self, trace: Trace, plan=None) -> FTExperiment:
        """Generate experiment with comprehensive parameter decisions."""
        
        # 1. Collect context information
        context = self._prepare_context(trace)
        
        # 2. Generate hypothesis using LLM
        hypothesis = self._generate_hypothesis_with_llm(context)
        
        # 3. Validate hypothesis (minimal check)
        is_valid, error_msg = self.llama_manager.validate_hypothesis_json(
            hypothesis.hypothesis_json
        )
        if not is_valid:
            logger.warning(f"Potential issue detected: {error_msg}")
            logger.info("Proceeding anyway, trusting LLM's judgment")
        
        logger.info(f"Generated hypothesis: {hypothesis.base_model} with {hypothesis.finetune_method}")
        
        # 4. Convert to experiment
        return FTHypothesis2Experiment().convert(hypothesis, trace)
    
    def _prepare_context(self, trace: Trace) -> Dict:
        """Prepare complete context for ExpGen"""
        
        # Get device and dataset information
        device_info = get_runtime_environment_by_env(get_ft_env())
        dataset_info = extract_dataset_info(FT_RD_SETTING.dataset)
        device_dict = json.loads(device_info)
        memory_gb = device_dict.get("gpu", {}).get("total_gpu_memory_gb")
        
        logger.info(f"Device: {memory_gb}GB GPU")
        logger.info(f"Dataset: {dataset_info['name']}")
        
        context = {
            "task": {
                "dataset": FT_RD_SETTING.dataset,
                "base_model": FT_RD_SETTING.base_model,
                "objective": FT_RD_SETTING.task,
                "gpu_memory_gb": memory_gb,
                "dataset_info": dataset_info,
            },
            
            # All available parameters (compact version)
            "parameters": self.llama_manager.get_compact_parameters_summary(),
            
            # Enum choices
            "choices": self.llama_manager.get_available_choices(),
            
            # Constraint rules
            "constraints": self.llama_manager.get_constraint_rules(),
        }
        
        # Add feedback from previous experiment
        previous_feedback = self._extract_previous_feedback(trace)
        if previous_feedback:
            context["feedback"] = previous_feedback
            logger.info(f"Including feedback from experiment #{previous_feedback['experiment_id']}")
        
        return context
    
    def _extract_previous_feedback(self, trace: Trace) -> Optional[Dict]:
        """Extract feedback from the last experiment in trace"""
        if not trace.hist or len(trace.hist) == 0:
            return None
        
        # Get most recent experiment
        last_exp = trace.hist[-1]
        if not hasattr(last_exp, "result") or not last_exp.result:
            return None
        
        # Extract key information
        feedback = {
            "experiment_id": len(trace.hist),
            "hypothesis_json": {},
            "status": "unknown",
        }
        
        # Get hypothesis JSON if available
        if hasattr(last_exp, "hypothesis") and hasattr(last_exp.hypothesis, "hypothesis_json"):
            feedback["hypothesis_json"] = last_exp.hypothesis.hypothesis_json
        
        # Determine status
        if hasattr(last_exp.result, "final_decision"):
            feedback["status"] = "success" if last_exp.result.final_decision else "failed"
        
        # Extract feedback content if available
        if hasattr(last_exp.result, "feedback"):
            fb = last_exp.result.feedback
            if hasattr(fb, "observations"):
                feedback["observations"] = fb.observations
            if hasattr(fb, "hypothesis_evaluation"):
                feedback["evaluation"] = fb.hypothesis_evaluation
            if hasattr(fb, "new_hypothesis"):
                feedback["suggestions"] = fb.new_hypothesis
        
        return feedback
    
    def _generate_hypothesis_with_llm(self, context: Dict) -> FTHypothesis:
        """Use LLM to generate hypothesis in JSON format"""
        
        try:
            system_prompt = T("scenarios.finetune.proposal.prompts:expgen.system").r(
                parameters_count=len(context["parameters"]),
                constraints="\n".join(f"- {rule}" for rule in context["constraints"]),
            )
            
            # Pass all parameters (already in compact format: name + type + default)
            user_prompt = T("scenarios.finetune.proposal.prompts:expgen.user").r(
                task=context["task"],
                parameters=context["parameters"],
                choices=json.dumps(context["choices"], indent=2),
                feedback=context.get("feedback"),
                has_feedback=("feedback" in context),
            )
            
            # Call LLM with JSON mode
            api = APIBackend()
            response = api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            )
            
            # Parse JSON response
            hypothesis_json = json.loads(response)
            
            # Construct hypothesis
            hypothesis = FTHypothesis(hypothesis_json=hypothesis_json)
            
            return hypothesis
            
        except Exception as e:
            logger.error(f"Failed to generate hypothesis with LLM: {e}")
            logger.warning("Using fallback hypothesis generation")
            return self._create_fallback_hypothesis(context)
    
    def _create_fallback_hypothesis(self, context: Dict) -> FTHypothesis:
        """Create fallback hypothesis in JSON format"""
        memory_gb = context["task"]["gpu_memory_gb"]
        
        # Generate JSON based on memory
        if memory_gb and memory_gb < 16:
            hypothesis_json = {
                "reasoning": f"Fallback configuration for {memory_gb}GB GPU: Using QLoRA with conservative settings for stability",
                "quantization_method": "bitsandbytes",
                "quantization_bit": 4,
                "quantization_type": "nf4",
                "double_quantization": True,
                "finetuning_type": "lora",
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "learning_rate": 5e-5,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.03,
                "cutoff_len": 1024,
            }
        else:
            hypothesis_json = {
                "reasoning": "Fallback configuration: Standard LoRA with conservative hyperparameters",
                "finetuning_type": "lora",
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "learning_rate": 5e-5,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.03,
                "cutoff_len": 1024,
            }
        
        return FTHypothesis(hypothesis_json=hypothesis_json)
