# LLaMA Factory Parameter Query Agent

A decision-based MCP (Model Context Protocol) agent for querying LLaMA Factory training parameters through intelligent filtering.

## Overview

This MCP implementation provides:
- **Decision-based filtering**: Reduces 270+ parameters to 30-70 relevant ones
- **Direct source code access**: Always up-to-date, no cache maintenance
- **Complete information**: No truncated help text
- **Natural language queries**: Flexible parameter exploration

## Installation

### 1. Install Dependencies

```bash
conda activate rdagent
pip install fastapi uvicorn[standard]
```

### 2. Verify LLaMA Factory Path

The MCP server auto-detects LLaMA Factory at:
```
git_ignore_folder/LLaMA-Factory/
```

If your path is different, set environment variable:
```bash
export LLAMA_FACTORY_PATH=/path/to/LLaMA-Factory
```

## Quick Start

### Step 1: Start MCP Server

#### Option A: Using Script (Recommended)

```bash
conda activate rdagent
cd /path/to/RD-Agent
./scripts/start_mcp_server.sh
```

#### Option B: Direct Command

```bash
conda activate rdagent
cd /path/to/RD-Agent
python -m rdagent.components.agent.mcp.servers.llama_factory_server
```

Expected output:
```
🚀 LLaMA Factory MCP Server
📁 Source: /path/to/LLaMA-Factory/src/llamafactory/hparams
🌐 Listening on http://0.0.0.0:8124
📊 Available files: 7
INFO:     Uvicorn running on http://0.0.0.0:8124 (Press CTRL+C to quit)
```

### Step 2: Test Agent (Optional)

```bash
# In another terminal
python3 << 'EOF'
from rdagent.components.agent.llama_factory import Agent

agent = Agent()
result = agent.query("List all parameters for LoRA fine-tuning")
print(result)
EOF
```

### Step 3: Run FT Scenario

```bash
# MCP is enabled by default in FT scenario
python -m rdagent.app.finetune.llm \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset alpaca
```

The agent will automatically query relevant parameters during ExpGen stage.

## Configuration

### Environment Variables

```bash
# MCP Server
export LLAMA_FACTORY_PATH=/path/to/LLaMA-Factory  # Auto-detected
export MCP_LLAMA_PORT=8124                          # Default: 8124

# MCP Client
export LLAMA_FACTORY_MCP_URL=http://localhost:8124/mcp  # Default
export LLAMA_FACTORY_MCP_TIMEOUT=60                      # Default: 60s

# FT Scenario
export FT_ENABLE_MCP_PARAM_SEARCH=false  # Disable if needed (enabled by default)
```

### Disable MCP (if needed)

```bash
# Option 1: Environment variable
export FT_ENABLE_MCP_PARAM_SEARCH=false

# Option 2: Modify conf.py
# rdagent/app/finetune/llm/conf.py
enable_mcp_param_search: bool = False
```

## Decision-Based Filtering

### 8 Decision Dimensions

The agent filters parameters based on training decisions:

```python
decisions = {
    "stage": "sft",           # sft/pt/rm/ppo/dpo/kto
    "method": "lora",         # lora/freeze/oft/full
    "quantization": "4bit",   # none/4bit/8bit
    "optimizer": "standard",  # standard/apollo/badam/galore
    "distributed": "single",  # single/ddp/deepspeed/fsdp
    "precision": "fp16",      # fp32/fp16/bf16/pure_bf16
    "data_strategy": "normal",# normal/streaming/packing
    "modality": "text"        # text/multimodal
}
```

### Filtering Effect

```
Original: 270+ parameters
After filtering: 30-70 parameters

Example (LoRA + 4bit + standard):
✅ Keep: LoraArguments (13 params)
✅ Keep: Core training params (10 params)
✅ Keep: Quantization params (2 params)
❌ Drop: FreezeArguments (3 params)
❌ Drop: OFTArguments (6 params)
❌ Drop: Apollo/BAdam/GaLore (25 params)
❌ Drop: RLHF params (21 params)
❌ Drop: Advanced training params (100+ params)
```

## Architecture

```
┌──────────────────────────────────────┐
│  FT ExpGen / Coder                   │
│  Decision: method=lora, quant=4bit   │
└──────────────┬───────────────────────┘
               │
               ↓
┌──────────────────────────────────────┐
│  LLaMA Factory Agent                 │
│  - System prompt (decision guidance) │
│  - MCP toolset                       │
└──────────────┬───────────────────────┘
               │ HTTP (MCP Protocol)
               ↓
┌──────────────────────────────────────┐
│  MCP Server (Port 8124)              │
│  Tools:                              │
│  1. list_parameter_files             │
│  2. read_source_file                 │
│  3. read_constants                   │
│  4. get_decision_guidance            │
└──────────────┬───────────────────────┘
               │ File System
               ↓
┌──────────────────────────────────────┐
│  LLaMA Factory Source Code           │
│  src/llamafactory/hparams/*.py       │
└──────────────────────────────────────┘
```

## Usage Examples

### Basic Query

```python
from rdagent.components.agent.llama_factory import Agent

agent = Agent()

# Simple parameter query
result = agent.query("What is lora_rank and its default value?")
print(result)
```

### Decision-Based Query

```python
# Query with specific configuration
result = agent.query("""
Get parameters for the following configuration:
- Method: lora
- Quantization: 4bit
- Optimizer: standard
- Stage: sft

Provide parameter details including names, types, defaults, and descriptions.
""")
print(result)
```

### Method Comparison

```python
# Compare different methods
result = agent.query("Compare LoRA vs Freeze fine-tuning parameters")
print(result)
```

### Full Configuration

```python
# Get all relevant parameters for complex setup
result = agent.query("""
Get parameters for:
- Method: lora
- Quantization: 4bit
- Optimizer: apollo
- Distributed: deepspeed
Include all relevant parameters grouped by category.
""")
print(result)
```

## Integration with FT Scenario

The agent is automatically used in ExpGen when enabled:

```python
# rdagent/scenarios/finetune/proposal/proposal.py
class LLMFinetuneExpGen(ExpGen):
    def __init__(self, scen):
        self.param_agent = None
        if FT_RD_SETTING.enable_mcp_param_search:
            try:
                from rdagent.components.agent.llama_factory import Agent
                self.param_agent = Agent()
                logger.info("LLaMA Factory MCP Agent initialized")
            except Exception as e:
                logger.warning(f"MCP Agent init failed: {e}, using static params")
    
    def gen(self, trace):
        if self.param_agent:
            # Query parameters based on decisions
            param_info = self.param_agent.query(f"""
            Get parameters for:
            - Method: {method}
            - Quantization: {quantization}
            - Optimizer: {optimizer}
            
            Include required, method-specific, and core training parameters.
            """)
```

## Troubleshooting

### Issue 1: Module Not Found

```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
```bash
conda activate rdagent
pip install fastapi uvicorn[standard]
```

### Issue 2: Connection Refused

```
ConnectionError: [Errno 111] Connection refused
```

**Solution:** Make sure MCP server is running
```bash
python -m rdagent.components.agent.mcp.servers.llama_factory_server
```

### Issue 3: LLaMA Factory Not Found

```
FileNotFoundError: LLaMA Factory hparams not found
```

**Solution:**
```bash
export LLAMA_FACTORY_PATH=/path/to/LLaMA-Factory
```

### Issue 4: Timeout

```
TimeoutError: MCP query timed out
```

**Solution:** Increase timeout
```bash
export LLAMA_FACTORY_MCP_TIMEOUT=120
```

### Issue 5: Port Already in Use

```
OSError: [Errno 98] Address already in use
```

**Solution:** Change port or kill existing process
```bash
export MCP_LLAMA_PORT=8125
# Or
lsof -ti:8124 | xargs kill -9
```

## Testing

### Unit Tests

```bash
# Make sure MCP server is running
python -m rdagent.components.agent.mcp.servers.llama_factory_server

# In another terminal
pytest test/agent/test_llama_factory_mcp.py -v -s
```

### Manual Testing

```bash
# Test 1: Simple query
python3 << 'EOF'
from rdagent.components.agent.llama_factory import Agent
agent = Agent()
print(agent.query("What is lora_rank?"))
EOF

# Test 2: Decision-based query
python3 << 'EOF'
from rdagent.components.agent.llama_factory import Agent
agent = Agent()
print(agent.query("""
Get parameters for: method=lora, quantization=4bit
"""))
EOF
```

## Advantages

| Feature | Traditional (JSON) | MCP Agent |
|---------|-------------------|-----------|
| **Parameters** | 270+ all at once | 30-70 filtered |
| **Updates** | Manual re-extraction | Auto (reads source) |
| **Information** | Truncated (100 chars) | Complete descriptions |
| **Flexibility** | Fixed structure | Dynamic queries |
| **Maintenance** | Update JSON cache | Zero maintenance |
| **Extension** | Hardcode mappings | Data-driven |

## Benefits

1. **Reduced Complexity**: 88% parameter reduction (270 → 30-70)
2. **Always Updated**: Reads source code directly, no stale cache
3. **Complete Information**: Full parameter descriptions, no truncation
4. **Intelligent Filtering**: 8-dimensional decision tree
5. **Zero Maintenance**: No JSON extraction or updates needed
6. **Flexible Queries**: Natural language interface for exploration
7. **Extensible**: New methods/optimizers automatically supported

## Implementation Details

### MCP Server

- **Location**: `rdagent/components/agent/mcp/servers/llama_factory_server.py`
- **Port**: 8124 (configurable)
- **Tools**: 4 (list files, read source, read constants, decision guidance)
- **Code**: ~330 lines (including comments)

### Agent Client

- **Location**: `rdagent/components/agent/llama_factory/__init__.py`
- **Base**: PAIAgent (pydantic-ai wrapper)
- **Protocol**: MCP over HTTP
- **Code**: ~45 lines

### Decision Mapping

```python
DECISION_MAP = {
    "lora": ["LoraArguments"],
    "freeze": ["FreezeArguments"],
    "oft": ["OFTArguments"],
    "apollo": ["ApolloArguments"],
    "badam": ["BAdamArgument"],
    "galore": ["GaloreArguments"],
    "rlhf": ["RLHFArguments"],
}
```

### Core Focus Parameters

```python
CORE_PARAMS = [
    # Required
    "model_name_or_path", "dataset", "dataset_dir", 
    "stage", "finetuning_type", "output_dir", "do_train",
    # Core training
    "learning_rate", "num_train_epochs", 
    "per_device_train_batch_size", "gradient_accumulation_steps",
    "warmup_steps", "lr_scheduler_type", "optim", "max_grad_norm",
]
```

## Workflow Example

```bash
# 1. Start MCP server (Terminal 1)
python -m rdagent.components.agent.mcp.servers.llama_factory_server

# 2. Run FT scenario (Terminal 2)
python -m rdagent.app.finetune.llm \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset alpaca

# What happens:
# ┌─────────────────────────────────────┐
# │ ExpGen Stage                        │
# ├─────────────────────────────────────┤
# │ 1. Decide: method=lora, quant=4bit  │
# │ 2. Query MCP Agent                  │
# │ 3. Get ~35 relevant parameters      │
# │ 4. Inject into coder prompt         │
# └─────────────────────────────────────┘
# ┌─────────────────────────────────────┐
# │ Coder Stage                         │
# ├─────────────────────────────────────┤
# │ 1. Generate train.yaml              │
# │ 2. Use filtered parameters          │
# │ 3. Better quality (less overload)   │
# └─────────────────────────────────────┘
```

## Performance

- **Parameter reduction**: 74% (270 → 70 max)
- **Typical reduction**: 88% (270 → 30-35 for common configs)
- **Query latency**: ~2-5 seconds (includes LLM processing)
- **Server startup**: <1 second
- **Memory footprint**: ~50MB (FastAPI server)

## Future Extensions

Potential enhancements:
- [ ] Cache frequently queried parameters (session-level)
- [ ] Support for custom parameter classes
- [ ] Multi-language support (currently English)
- [ ] Integration with WebUI for interactive parameter exploration
- [ ] Automatic parameter validation against LLaMA Factory version

## Contributing

When adding new decision dimensions or parameter classes:

1. Update `DECISION_MAP` in `llama_factory_server.py`
2. Update decision guidance logic in `get_decision_guidance()`
3. Update documentation (this README)
4. Add test cases

## References

- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
- [MCP Protocol](https://github.com/pydantic/pydantic-ai)
- [Pydantic AI](https://ai.pydantic.dev/)
- [FastAPI](https://fastapi.tiangolo.com/)
