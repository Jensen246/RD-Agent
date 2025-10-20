"""
LLaMA Factory MCP Server - Direct Introspection Edition

A minimal MCP server that directly introspects LLaMA Factory's dataclass
structure to expose all training parameters and capabilities.

Design Philosophy:
    "Talk is cheap. Show me the code." - Linus Torvalds
    
    - Zero hardcoding: Use Python's dataclass reflection
    - Zero caching: Import and reflect on demand
    - Zero filtering: Give the agent ALL parameters, let it decide

Usage:
    python -m rdagent.components.agent.mcp.servers.llama_factory_server

Environment Variables:
    MCP_LLAMA_PORT: Server port (default: 8125)
"""

import json
import sys
from dataclasses import fields, MISSING
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn


# Add LLaMA Factory to path
LLAMA_FACTORY_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "git_ignore_folder" / "LLaMA-Factory" / "src"
if LLAMA_FACTORY_PATH.exists():
    sys.path.insert(0, str(LLAMA_FACTORY_PATH))


class SimpleLLaMAFactoryIntrospector:
    """
    Direct introspection of LLaMA Factory's dataclass structure.
    No caching, no filtering, just raw reflection.
    """
    
    def get_parameter_schema(self, category: str = "all") -> Dict[str, Any]:
        """
        Extract complete parameter schema from dataclasses.
        
        Args:
            category: "training" | "finetuning" | "model" | "data" | "all"
        
        Returns:
            Schema dict with parameters, types, defaults, and source classes
        """
        from llamafactory.hparams import (
            TrainingArguments,
            FinetuningArguments,
            ModelArguments,
            DataArguments,
        )
        
        schemas = {}
        
        if category in ["training", "all"]:
            schemas["training"] = self._extract_schema(TrainingArguments)
        
        if category in ["finetuning", "all"]:
            schemas["finetuning"] = self._extract_schema(FinetuningArguments)
        
        if category in ["model", "all"]:
            schemas["model"] = self._extract_schema(ModelArguments)
        
        if category in ["data", "all"]:
            schemas["data"] = self._extract_schema(DataArguments)
        
        return schemas
    
    def _extract_schema(self, cls) -> Dict[str, Any]:
        """
        Extract schema from a single dataclass.
        """
        schema = {
            "class_name": cls.__name__,
            "inheritance_chain": [c.__name__ for c in cls.__mro__ if c.__name__ not in ["object"]],
            "parameters": {},
        }
        
        for field in fields(cls):
            # Skip internal fields
            if field.name.startswith("_"):
                continue
            
            schema["parameters"][field.name] = {
                "type": self._format_type(field.type),
                "default": self._format_default(field.default, field.default_factory),
                "description": field.metadata.get("help", ""),
                "source_class": self._find_source_class(cls, field.name),
            }
        
        schema["total_params"] = len(schema["parameters"])
        return schema
    
    def _format_type(self, type_annotation) -> str:
        """Format type annotation as readable string."""
        if hasattr(type_annotation, "__name__"):
            return type_annotation.__name__
        return str(type_annotation).replace("typing.", "")
    
    def _format_default(self, default, default_factory) -> Any:
        """Format default value."""
        if default is not MISSING:
            # Handle special cases
            if default is None:
                return None
            if isinstance(default, (str, int, float, bool)):
                return default
            return str(default)
        elif default_factory is not MISSING:
            return "<factory>"
        return "<required>"
    
    def _find_source_class(self, cls, field_name: str) -> str:
        """Find which class in the inheritance chain defines this field."""
        for base in cls.__mro__:
            if hasattr(base, "__dataclass_fields__") and field_name in base.__dataclass_fields__:
                return base.__name__
        return cls.__name__
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get supported capabilities from constants and transformers enums.
        """
        try:
            from llamafactory.extras.constants import METHODS, TRAINING_STAGES
        except ImportError as e:
            # Fallback to hardcoded values if constants not available
            METHODS = ["lora", "freeze", "full"]
            TRAINING_STAGES = ["pt", "sft", "rm", "ppo", "dpo", "kto"]
        
        try:
            from transformers.training_args import OptimizerNames, SchedulerType
            optimizers = self._extract_enum_values(OptimizerNames)
            schedulers = self._extract_enum_values(SchedulerType)
        except ImportError as e:
            # Fallback to common values
            optimizers = ["adamw_torch", "adamw_8bit", "sgd", "adafactor"]
            schedulers = ["linear", "cosine", "constant", "polynomial"]
        
        return {
            "methods": list(METHODS) if hasattr(METHODS, "__iter__") else METHODS,
            "stages": list(TRAINING_STAGES) if hasattr(TRAINING_STAGES, "__iter__") else TRAINING_STAGES,
            "optimizers": optimizers,
            "schedulers": schedulers,
        }
    
    def _extract_enum_values(self, enum_class) -> List[str]:
        """Extract all values from an enum or enum-like class."""
        values = []
        for name in dir(enum_class):
            if not name.startswith("_"):
                attr = getattr(enum_class, name)
                if isinstance(attr, str):
                    values.append(attr)
        return values


# ============ MCP Server ============

app = FastAPI(title="LLaMA Factory MCP Server")
app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
introspector = SimpleLLaMAFactoryIntrospector()


@app.post("/")
async def handle_mcp_request(request: Request):
    """Handle MCP protocol requests."""
    body = await request.json()
    method = body.get("method")
    
    if method == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "protocolVersion": "2025-06-18",
                "serverInfo": {
                    "name": "llama-factory-server",
                    "version": "3.0.0"
                },
                "capabilities": {
                    "tools": {}
                }
            }
        }
    elif method == "tools/list":
        response = {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "tools": [
                    {
                        "name": "get_parameter_schema",
                        "description": "Get complete parameter schema for LLaMA Factory. Returns all parameters with types, defaults, descriptions, and source classes.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "description": "Parameter category: 'training', 'finetuning', 'model', 'data', or 'all' (default)",
                                    "enum": ["training", "finetuning", "model", "data", "all"],
                                    "default": "all"
                                }
                            }
                        }
                    },
                    {
                        "name": "get_capabilities",
                        "description": "Get supported methods, stages, optimizers, and schedulers. Returns lists of all available options.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        tool_name = body.get("params", {}).get("name")
        arguments = body.get("params", {}).get("arguments", {})
        
        try:
            if tool_name == "get_parameter_schema":
                category = arguments.get("category", "all")
                schema = introspector.get_parameter_schema(category)
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(schema, indent=2, ensure_ascii=False)
                        }
                    ]
                }
            elif tool_name == "get_capabilities":
                capabilities = introspector.get_capabilities()
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(capabilities, indent=2, ensure_ascii=False)
                        }
                    ]
                }
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
            
            response = {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": result
            }
        except Exception as e:
            response = {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }
    else:
        response = {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }
    
    return response


def main():
    """Start the MCP server."""
    port = int(os.environ.get("MCP_LLAMA_PORT", 8125))
    print(f"Starting LLaMA Factory MCP Server on port {port}")
    print(f"LLaMA Factory path: {LLAMA_FACTORY_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import os
    main()
