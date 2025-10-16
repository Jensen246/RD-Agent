"""
LLaMA Factory Parameter Manager

Responsibilities:
1. Manage parameters extracted from LLaMA Factory
2. Provide convenient API for accessing hierarchical parameters
3. Provide parameter summaries for ExpGen and Coder
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger


class LLaMAFactoryManager:
    """LLaMA Factory Parameter Manager"""

    def __init__(self):
        """Initialize manager instance"""
        base_path = FT_RD_SETTING.file_path
        self.cache_dir = Path(base_path) / ".llama_factory_info"
        self._info_cache: Optional[Dict] = None
        self.update_llama_factory = FT_RD_SETTING.update_llama_factory

    def extract_info_from_docker(self) -> Dict:
        """Extract LLaMA Factory information from Docker environment"""
        if self.update_llama_factory or not self.cache_dir.exists() or not any(self.cache_dir.iterdir()):
            logger.info("Update & Extract LLaMA Factory parameters from Docker")
            # Prepare extraction script
            workspace = FBWorkspace()
            script_path = Path(__file__).parent / "docker_scripts" / "pull_extract.py"
            workspace.inject_files(**{"extract_script.py": script_path.read_text()})

            # Setup cache directory and Docker volumes
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            volumes = {str(self.cache_dir): {"bind": "/workspace/.llama_factory_info", "mode": "rw"}}

            # Run extraction
            result = workspace.run(
                env=get_ft_env(extra_volumes=volumes, running_timeout_period=120, enable_cache=False),
                entry="python extract_script.py",
            )

            if result.exit_code != 0:
                raise RuntimeError(f"Parameter extraction failed: {result.stdout}")

        else:
            logger.info("Skip updating LLaMA Factory, using local cache")

        # Load extracted data
        self._info_cache = self._load_extracted_data()
        if not self._info_cache:
            raise RuntimeError("Failed to load extracted LLaMA Factory information")

        logger.info("Successfully extracted LLaMA Factory parameters")
        return self._info_cache

    def _load_extracted_data(self) -> Dict:
        """Load extracted parameter information"""
        data = {}

        # Load constants
        constants_file = self.cache_dir / "constants.json"
        if constants_file.exists():
            with open(constants_file, encoding="utf-8") as f:
                data.update(json.load(f))

        # Load parameters
        parameters_file = self.cache_dir / "parameters.json"
        if parameters_file.exists():
            with open(parameters_file, encoding="utf-8") as f:
                data["parameters"] = json.load(f)

        # Load metadata (optional)
        metadata_file = self.cache_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, encoding="utf-8") as f:
                data["metadata"] = json.load(f)

        return data

    def get_info(self) -> Dict:
        """Get complete LLaMA Factory information"""
        if self._info_cache is None:
            self._info_cache = self.extract_info_from_docker()
        return self._info_cache

    @property
    def methods(self) -> List[str]:
        """Available fine-tuning methods"""
        return self.get_info().get("methods", [])

    @property
    def models(self) -> List[str]:
        """Available base models"""
        return list(self.get_info().get("supported_models", {}).keys())

    @property
    def hf_models(self) -> List[str]:
        """Available HuggingFace models"""
        supported_models = self.get_info().get("supported_models", {})
        hf_model_set = set()
        for hf_model in supported_models.values():
            if isinstance(hf_model, str):
                hf_model_set.add(hf_model)
        return list(hf_model_set)

    @property
    def peft_methods(self) -> List[str]:
        """Available PEFT methods, filtered from methods list"""
        known_peft = {"lora", "qlora", "adalora"}
        return [m for m in self.methods if m in known_peft]

    @property
    def training_stages(self) -> Dict[str, str]:
        """Training stage mapping"""
        return self.get_info().get("training_stages", {})

    @property
    def templates(self) -> List[str]:
        """Available chat templates"""
        return self.get_info().get("templates", [])

    def get_template_for_model(self, model_name: str) -> Optional[str]:
        """Get model template (returns None to let LlamaFactory auto-detect)"""
        return None

    def is_peft_method(self, method: str) -> bool:
        """Check if method is a PEFT method"""
        return method in self.peft_methods

    def get_parameters(self, param_type: Optional[str] = None) -> Dict:
        """Get parameters (by type or all)"""
        params = self.get_info().get("parameters", {})
        if param_type:
            return params.get(param_type, {})
        return params

    def get_method_specific_params(self, method: str) -> Dict:
        """Get method-specific parameters
        
        Automatically find corresponding Arguments class from finetuning parameter hierarchy.
        
        Args:
            method: Fine-tuning method name (e.g., "lora", "freeze")
            
        Returns:
            Parameter dictionary for the method, empty dict if not found
        """
        finetuning_params = self.get_parameters("finetuning")
        
        # Find matching Arguments class
        # e.g., method="lora" → find "LoraArguments"
        method_key = f"{method.capitalize()}Arguments"
        
        if method_key in finetuning_params:
            return finetuning_params[method_key]
        
        # Fallback: case-insensitive search
        for key, params in finetuning_params.items():
            if key.lower().startswith(method.lower()):
                return params
        
        return {}

    # ========== NEW METHODS FOR EXPGEN/CODER SEPARATION ==========
    
    def get_compact_parameters_summary(self) -> Dict[str, str]:
        """Get compact parameter summary for ExpGen (name + type + default, no help text)
        
        This provides high information density (~1700 tokens for 334 parameters).
        Format: "category.name: type = default"
        
        Returns:
            Dict mapping parameter keys to compact signatures
            {
                "model.QuantizationArguments.quantization_bit": "Optional[int] = None",
                "finetuning.LoraArguments.lora_rank": "int = 8",
                "training.learning_rate": "float = 5e-5",
                ...
            }
        """
        summary = {}
        params = self.get_info().get("parameters", {})
        
        for category, content in params.items():
            if isinstance(content, dict):
                # Handle flat structure (data, training)
                if all(isinstance(v, dict) and "type" in v for v in content.values()):
                    for param_name, param_info in content.items():
                        key = f"{category}.{param_name}"
                        summary[key] = self._format_param_signature(param_info)
                
                # Handle hierarchical structure (model, finetuning)
                else:
                    for sub_category, sub_params in content.items():
                        if isinstance(sub_params, dict):
                            for param_name, param_info in sub_params.items():
                                if isinstance(param_info, dict) and "type" in param_info:
                                    # Flatten: model.QuantizationArguments.quantization_bit
                                    key = f"{category}.{sub_category}.{param_name}"
                                    summary[key] = self._format_param_signature(param_info)
        
        return summary
    
    def _format_param_signature(self, param_info: Dict) -> str:
        """Format parameter signature as compact string (type + default)"""
        param_type = param_info.get("type", "Any")
        default = param_info.get("default")
        
        if default is None:
            return f"Optional[{param_type}] = None"
        elif default == "<factory>":
            return f"{param_type} = <default>"
        elif isinstance(default, str):
            return f"{param_type} = '{default}'"
        else:
            return f"{param_type} = {default}"
    
    def get_available_choices(self) -> Dict[str, List]:
        """Get available choices for enum-type parameters"""
        return {
            "finetuning_type": self.methods,
            "stage": list(self.training_stages.keys()),
            "quantization_method": ["bitsandbytes", "gptq", "hqq"],
            "template": self.templates[:20],  # Limit to common ones
        }
    
    def get_constraint_rules(self) -> List[str]:
        """Get parameter constraint rules (only technical limitations from LlamaFactory)"""
        return [
            "LoRA/QLoRA cannot be used with GaLore/BAdam/Apollo optimizers (LlamaFactory limitation)",
            "Can only use one optimizer: GaLore OR BAdam OR Apollo are mutually exclusive",
        ]
    
    def get_relevant_parameters_with_help(self, decisions: Dict) -> Dict[str, Any]:
        """Get relevant parameters with full help text for Coder (filtered based on ExpGen decisions)
        
        Args:
            decisions: ExpGen's parameter decisions, format:
                {
                    "model": {"quantization_bit": 4, ...},
                    "finetuning": {"finetuning_type": "lora", "lora_rank": 8, ...},
                    "training": {"learning_rate": 1e-4, ...},
                    "data": {...}
                }
        
        Returns:
            Filtered parameters with help text, only including relevant categories
        """
        relevant_params = {}
        params = self.get_info().get("parameters", {})
        
        # Always include data and training (core categories)
        for category in ["data", "training"]:
            if category in params:
                relevant_params[category] = params[category]
        
        # Handle model parameters
        if "model" in decisions and decisions["model"]:
            model_params = params.get("model", {})
            relevant_params["model"] = {}
            
            # If quantization is used, include QuantizationArguments
            if "quantization_bit" in decisions["model"] or "quantization_method" in decisions["model"]:
                if "QuantizationArguments" in model_params:
                    relevant_params["model"]["QuantizationArguments"] = model_params["QuantizationArguments"]
            
            # Always include base model parameters
            if "BaseModelArguments" in model_params:
                relevant_params["model"]["BaseModelArguments"] = model_params["BaseModelArguments"]
            
            # Include _own parameters if exist
            if "_own" in model_params:
                relevant_params["model"]["_own"] = model_params["_own"]
        
        # Handle finetuning parameters
        if "finetuning" in decisions and decisions["finetuning"]:
            finetuning_params = params.get("finetuning", {})
            relevant_params["finetuning"] = {}
            
            # Include base finetuning parameters
            if "_own" in finetuning_params:
                relevant_params["finetuning"]["_own"] = finetuning_params["_own"]
            
            # Include method-specific parameters based on finetuning_type
            finetuning_type = decisions["finetuning"].get("finetuning_type")
            if finetuning_type:
                method_key = f"{finetuning_type.capitalize()}Arguments"
                if method_key in finetuning_params:
                    relevant_params["finetuning"][method_key] = finetuning_params[method_key]
            
            # Include optimizer if used
            if decisions["finetuning"].get("use_galore") and "GaloreArguments" in finetuning_params:
                relevant_params["finetuning"]["GaloreArguments"] = finetuning_params["GaloreArguments"]
            
            if decisions["finetuning"].get("use_badam") and "BAdamArgument" in finetuning_params:
                relevant_params["finetuning"]["BAdamArgument"] = finetuning_params["BAdamArgument"]
            
            if decisions["finetuning"].get("use_apollo") and "ApolloArguments" in finetuning_params:
                relevant_params["finetuning"]["ApolloArguments"] = finetuning_params["ApolloArguments"]
            
            # Include RLHF parameters if stage is not sft
            stage = decisions["finetuning"].get("stage", "sft")
            if stage in ["dpo", "ppo", "kto"] and "RLHFArguments" in finetuning_params:
                relevant_params["finetuning"]["RLHFArguments"] = finetuning_params["RLHFArguments"]
        
        return relevant_params
    
    def validate_hypothesis_json(self, hypothesis_json: Dict) -> Tuple[bool, Optional[str]]:
        """Validate hypothesis JSON (minimal check, only detectable technical conflicts)
        
        Args:
            hypothesis_json: JSON hypothesis from ExpGen
            
        Returns:
            (is_valid, error_message)
            Note: Returns True if cannot detect issues (trust LLM)
        """
        if not hypothesis_json:
            return True, None
        
        # Helper function to search for keys recursively
        def find_in_dict(d: Any, keys: List[str]) -> Any:
            if not isinstance(d, dict):
                return None
            for key in keys:
                if key in d:
                    return d[key]
            for value in d.values():
                if isinstance(value, dict):
                    result = find_in_dict(value, keys)
                    if result is not None:
                        return result
            return None
        
        # Find relevant parameters
        ft_type = find_in_dict(hypothesis_json, ["finetuning_type", "method", "finetune_method", "ft_type"])
        use_galore = find_in_dict(hypothesis_json, ["use_galore", "galore"])
        use_badam = find_in_dict(hypothesis_json, ["use_badam", "badam"])
        use_apollo = find_in_dict(hypothesis_json, ["use_apollo", "apollo"])
        
        # Rule 1: LoRA + Optimizer conflict
        if ft_type == "lora" and (use_galore or use_badam or use_apollo):
            return False, "LoRA cannot be used with GaLore/BAdam/Apollo optimizers (LlamaFactory limitation)"
        
        # Rule 2: Multiple optimizers
        optimizer_count = sum([bool(use_galore), bool(use_badam), bool(use_apollo)])
        if optimizer_count > 1:
            return False, "Can only use one optimizer (GaLore/BAdam/Apollo are mutually exclusive)"
        
        # If we can't detect issues, trust the LLM
        return True, None


# Module-level singleton instance
_manager_instance: Optional[LLaMAFactoryManager] = None


def get_llama_factory_manager() -> LLaMAFactoryManager:
    """Get singleton LLaMAFactoryManager instance"""
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = LLaMAFactoryManager()

    return _manager_instance
