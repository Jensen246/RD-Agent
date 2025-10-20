# LLaMA Factory MCP Server (v3.0)

> "Talk is cheap. Show me the code." - Linus Torvalds

A minimal MCP server that directly introspects LLaMA Factory's dataclass structure to expose ALL training parameters and capabilities.

## Design Philosophy

**Zero Hardcoding, Zero Filtering**

1. **Direct Reflection**: Use Python's `dataclasses.fields()` to extract all parameters
2. **No Caching**: Import LLaMA Factory modules on demand
3. **No Filtering**: Return ALL parameters, let the agent decide what's relevant
4. **Inheritance Transparency**: Show which parameters come from base classes (e.g., `Seq2SeqTrainingArguments`)

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  SimpleLLaMAFactoryIntrospector                        │
├────────────────────────────────────────────────────────┤
│  • get_parameter_schema(category)                      │
│    └─> Returns ALL parameters with:                    │
│        - Type annotations                              │
│        - Default values                                │
│        - Documentation strings                         │
│        - Source class (inheritance tracking)           │
│                                                         │
│  • get_capabilities()                                  │
│    └─> Returns supported:                              │
│        - Methods: lora, freeze, full                   │
│        - Stages: pt, sft, dpo, ppo, kto, rm           │
│        - Optimizers: 43+ from transformers             │
│        - Schedulers: 11 from transformers              │
└────────────────────────────────────────────────────────┘
```

## API

### Tool 1: `get_parameter_schema`

Get complete parameter schema for training.

**Input:**
```json
{
  "category": "all"  // "training" | "finetuning" | "model" | "data" | "all"
}
```

**Output:**
```json
{
  "training": {
    "class_name": "TrainingArguments",
    "inheritance_chain": ["TrainingArguments", "RayArguments", "Seq2SeqTrainingArguments", ...],
    "total_params": 138,
    "parameters": {
      "learning_rate": {
        "type": "float",
        "default": "5e-05",
        "description": "The initial learning rate for AdamW.",
        "source_class": "Seq2SeqTrainingArguments"
      },
      "lora_rank": {
        "type": "int",
        "default": "8",
        "description": "The intrinsic dimension for LoRA tuning.",
        "source_class": "LoraArguments"
      },
      ...
    }
  },
  "finetuning": { ... },
  "model": { ... },
  "data": { ... }
}
```

### Tool 2: `get_capabilities`

Get supported methods, stages, optimizers, and schedulers.

**Input:**
```json
{}
```

**Output:**
```json
{
  "methods": ["lora", "freeze", "full"],
  "stages": ["pt", "sft", "rm", "ppo", "dpo", "kto"],
  "optimizers": [
    "adamw_torch", "adamw_8bit", "sgd", "lion", 
    "galore_adamw", "apollo_adamw", ...
  ],
  "schedulers": [
    "linear", "cosine", "polynomial", "constant_with_warmup", ...
  ]
}
```

## Parameter Statistics

| Category | Count | Source |
|----------|-------|--------|
| Training | 138+ | `Seq2SeqTrainingArguments` + `RayArguments` |
| Finetuning | 80+ | LoRA, GaLore, Apollo, BAdam, RLHF, ... |
| Model | 50+ | Model loading, quantization, export, ... |
| Data | 30+ | Dataset config, preprocessing, ... |
| **Total** | **298+** | All parameters exposed |

## Key Features

### 1. Complete Coverage

- **All 138 parameters** from `transformers.Seq2SeqTrainingArguments`
- **43+ optimizers** from transformers (ADAMW_TORCH, SGD, LION, GALORE_*, APOLLO_*, ...)
- **11 schedulers** from transformers (LINEAR, COSINE, POLYNOMIAL, ...)
- **All LLaMA Factory extensions** (LoRA, GaLore, Apollo, BAdam, RLHF, ...)

### 2. Source Tracking

Every parameter shows its source class:
- `learning_rate` → `Seq2SeqTrainingArguments` (from transformers)
- `lora_rank` → `LoraArguments` (from LLaMA Factory)
- `use_galore` → `GaloreArguments` (from LLaMA Factory)

### 3. Zero Maintenance

No JSON caches to update. No hardcoded lists. Just direct reflection.

## Usage Example

```python
from rdagent.components.agent.mcp.servers.llama_factory_server import SimpleLLaMAFactoryIntrospector

introspector = SimpleLLaMAFactoryIntrospector()

# Get all capabilities
caps = introspector.get_capabilities()
print(f"Supported optimizers: {caps['optimizers']}")

# Get all parameters
schema = introspector.get_parameter_schema("all")
print(f"Total parameters: {sum(s['total_params'] for s in schema.values())}")

# Get just finetuning parameters
finetuning = introspector.get_parameter_schema("finetuning")
print(f"LoRA rank default: {finetuning['finetuning']['parameters']['lora_rank']['default']}")
```

## Running the Server

```bash
# Start MCP server
python -m rdagent.components.agent.mcp.servers.llama_factory_server

# Or specify port
MCP_LLAMA_PORT=8125 python -m rdagent.components.agent.mcp.servers.llama_factory_server
```

## Agent Workflow (2-3 Rounds)

**Round 1: Explore Capabilities**
```
Agent calls: get_capabilities()
Result: Lists all methods, stages, optimizers, schedulers
Agent learns: "I can use LoRA, and there are 43 optimizer options"
```

**Round 2: Get Parameter Details**
```
Agent calls: get_parameter_schema("all")
Result: Complete schema with 298+ parameters
Agent reviews: Types, defaults, descriptions for relevant parameters
Agent generates hypothesis: "Use LoRA with rank=16, adamw_torch optimizer, cosine scheduler"
```

**Round 3: Generate Code**
```
Coder phase: Creates YAML config based on hypothesis and schema
```

## Code Size

- **v1.0**: 676 lines (complex filtering logic)
- **v2.0**: 450 lines (reduced hardcoding)
- **v3.0**: 286 lines (direct reflection, zero filtering)

**58% reduction in code complexity while providing 100% coverage.**

## Why This Works

### Old Approach (v1-v2)
```python
# Hardcode parameter importance
HIGH_IMPORTANCE = ["learning_rate", "lora_rank", ...]  # Endless maintenance

# Filter parameters
if decisions["method"] == "lora":
    return lora_params  # Agent sees limited context
```

### New Approach (v3)
```python
# Just reflect and return everything
from dataclasses import fields
return {p.name: extract_info(p) for p in fields(cls)}
```

**The agent is smart enough to handle 298 parameters. Stop trying to "help" it.**

## Dependencies

- `llamafactory` - For parameter definitions
- `transformers` - For base training arguments
- `fastapi` - For MCP server
- `uvicorn` - For running server

## License

Apache 2.0
