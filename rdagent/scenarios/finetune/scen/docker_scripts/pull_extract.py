"""
LLaMA Factory参数提取脚本 - 自动发现版本

设计原则：
1. 利用Python类型系统自动发现继承关系，不硬编码类名
2. 保留LLaMA Factory官方的参数组织结构
3. 类名即分组标签，不自作聪明重命名
"""

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

# Pull latest LLaMA Factory code first
try:
    result = subprocess.run(
        ["git", "pull", "--rebase"],
        cwd="/llamafactory",
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        output = (result.stdout or result.stderr).strip()
        print(f"Updated LLaMA Factory: {output}", file=sys.stderr)
    else:
        print(f"Warning: git pull failed: {result.stderr}", file=sys.stderr)
except Exception as e:
    print(f"Warning: Failed to update LLaMA Factory: {e}", file=sys.stderr)

# Add LLaMA Factory to path
sys.path.insert(0, "/llamafactory/src")

# Now import after updating
from llamafactory.data.template import TEMPLATES
from llamafactory.extras.constants import METHODS, SUPPORTED_MODELS, TRAINING_STAGES
from llamafactory.hparams import (
    DataArguments,
    FinetuningArguments,
    ModelArguments,
    TrainingArguments,
)


def extract_field_info(field):
    """提取单个字段的信息"""
    from dataclasses import MISSING

    # 处理默认值
    if hasattr(field, "default") and field.default is not MISSING:
        default_value = field.default
    elif hasattr(field, "default_factory") and field.default_factory is not MISSING:
        default_value = "<factory>"
    else:
        default_value = None

    return {
        "name": field.name,
        "type": str(field.type).replace("typing.", "").replace("<class '", "").replace("'>", ""),
        "default": default_value,
        "help": field.metadata.get("help", "") if field.metadata else "",
    }


def extract_hierarchy(cls):
    """自动提取类的继承层次结构
    
    利用Python的__mro__自动发现所有基类，不需要硬编码。
    每个基类的参数独立存储，保留官方的类名作为key。
    
    Args:
        cls: 要提取的dataclass
        
    Returns:
        dict: {
            "_own": {...},              # 这个类自己定义的参数
            "BaseClass1": {...},        # 第一个基类的参数
            "BaseClass2": {...},        # 第二个基类的参数
            ...
        }
    """
    result = {}
    
    # 获取所有字段名（包括继承的）
    all_field_names = {f.name for f in fields(cls)}
    
    # 遍历MRO（Method Resolution Order），自动发现所有基类
    for base_cls in cls.__mro__:
        # 跳过object和本类
        if base_cls in (object, cls):
            continue
        
        # 只处理dataclass
        if not hasattr(base_cls, '__dataclass_fields__'):
            continue
        
        # 获取这个基类定义的字段
        base_fields = base_cls.__dataclass_fields__
        
        # 提取字段信息
        base_params = {}
        for field_name, field_obj in base_fields.items():
            if field_name in all_field_names:
                # 从cls获取最新的field对象（可能被子类修改了）
                current_field = next((f for f in fields(cls) if f.name == field_name), None)
                if current_field:
                    base_params[field_name] = extract_field_info(current_field)
        
        if base_params:
            # 使用类名作为key，保留官方的命名
            result[base_cls.__name__] = base_params
    
    # 提取本类自己定义的参数（不在任何基类中的）
    base_field_names = set()
    for base_cls in cls.__mro__[1:]:  # 跳过自己
        if hasattr(base_cls, '__dataclass_fields__'):
            base_field_names.update(base_cls.__dataclass_fields__.keys())
    
    own_params = {}
    for field in fields(cls):
        if field.name not in base_field_names:
            own_params[field.name] = extract_field_info(field)
    
    if own_params:
        result["_own"] = own_params
    
    return result


def extract_flat(cls):
    """提取类的所有参数（扁平结构，用于简单的类）
    
    对于没有复杂继承关系的类（如DataArguments），
    使用扁平结构更简单。
    """
    return {field.name: extract_field_info(field) for field in fields(cls)}


def save_parameters(base_dir):
    """提取并保存LLaMA Factory参数
    
    设计原则：
    1. 自动发现继承关系（不硬编码类名）
    2. 保留官方的类名作为分组标签
    3. 对简单类使用扁平结构，对复杂类保留层次
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # 保存常量
    constants = {
        "methods": list(METHODS),
        "training_stages": dict(TRAINING_STAGES),
        "supported_models": dict(SUPPORTED_MODELS) if SUPPORTED_MODELS else {},
        "templates": list(TEMPLATES.keys()),
    }
    (base_path / "constants.json").write_text(json.dumps(constants, indent=2))

    # 保存参数
    # 设计说明：
    # - DataArguments: 简单类，无复杂继承，使用扁平结构
    # - TrainingArguments: 主要来自HuggingFace，使用扁平结构
    # - ModelArguments: 多重继承组合，保留层次结构
    # - FinetuningArguments: 多重继承组合，保留层次结构
    parameters = {
        "data": extract_flat(DataArguments),
        "training": extract_flat(TrainingArguments),
        "model": extract_hierarchy(ModelArguments),
        "finetuning": extract_hierarchy(FinetuningArguments),
    }
    
    (base_path / "parameters.json").write_text(json.dumps(parameters, indent=2))
    
    # 保存元信息，方便调试
    metadata = {
        "extraction_method": "automatic_hierarchy_discovery",
        "note": "Parameter structure automatically discovered from LLaMA Factory's class hierarchy",
        "model_bases": [cls.__name__ for cls in ModelArguments.__mro__ if hasattr(cls, '__dataclass_fields__')],
        "finetuning_bases": [cls.__name__ for cls in FinetuningArguments.__mro__ if hasattr(cls, '__dataclass_fields__')],
    }
    (base_path / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    """Main entry point for parameter extraction."""
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/.llama_factory_info"

    try:
        save_parameters(base_dir)
        print("Successfully extracted LLaMA Factory parameters")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
