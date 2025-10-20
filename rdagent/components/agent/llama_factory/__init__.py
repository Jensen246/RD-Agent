"""
LLaMA Factory Parameter Query Agent

A specialized agent for querying LLaMA Factory parameter definitions
through MCP (Model Context Protocol).

Usage:
    from rdagent.components.agent.llama_factory import Agent
    
    agent = Agent()
    
    # Query parameters for specific configuration
    result = agent.query('''
    Get parameters for:
    - Method: lora
    - Quantization: 4bit
    - Optimizer: standard
    ''')
    
    # Search specific parameter
    result = agent.query("What is lora_rank and its default value?")
"""

from pydantic_ai.mcp import MCPServerStreamableHTTP

from rdagent.components.agent.base import PAIAgent
from rdagent.components.agent.mcp.llama_factory import SETTINGS
from rdagent.utils.agent.tpl import T


class Agent(PAIAgent):
    """
    LLaMA Factory parameter query agent.
    
    This agent can:
    - Read LLaMA Factory source code
    - Provide parameter filtering guidance based on training decisions
    - Help understand parameter definitions and constraints
    """
    
    def __init__(self):
        toolsets = [MCPServerStreamableHTTP(SETTINGS.url, timeout=SETTINGS.timeout)]
        super().__init__(
            system_prompt=T(".prompts:system_prompt").r(),
            toolsets=toolsets
        )
    
    def query(self, query: str) -> str:
        """
        Query LLaMA Factory parameter information.
        
        Parameters
        ----------
        query : str
            Natural language query about parameters, e.g.:
            - "List parameters for LoRA fine-tuning"
            - "What is lora_alpha?"
            - "Get parameters for: method=lora, quantization=4bit"
        
        Returns
        -------
        str
            Parameter information with names, types, defaults, and descriptions
        """
        return super().query(query)

