"""
Test LLaMA Factory MCP Agent

Prerequisites:
    1. MCP server must be running on port 8124:
       python -m rdagent.components.agent.mcp.servers.llama_factory_server
    
    2. Run tests:
       pytest test/oai/test_mcp.py -v -s

Author: Test suite for LLaMA Factory parameter query agent
"""
import dotenv
dotenv.load_dotenv('.env')

import pytest
from rdagent.components.agent.llama_factory import Agent


class TestLlamaFactoryMCPAgent:
    """Test LLaMA Factory MCP Agent basic functionality"""

    @pytest.fixture(scope="class")
    def agent(self):
        """Create agent instance once for all tests"""
        try:
            agent = Agent()
            return agent
        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

    def test_agent_initialization(self, agent):
        """Test agent can be initialized"""
        assert agent is not None
        assert hasattr(agent, "query")

    def test_simple_parameter_query(self, agent):
        """Test querying a specific parameter"""
        result = agent.query("What is lora_rank and its default value?")
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain relevant information
        assert "lora" in result.lower() or "rank" in result.lower()

    def test_decision_based_query(self, agent):
        """Test querying with training decisions"""
        query = """
        Get parameters for the following configuration:
        - Method: lora
        - Quantization: 4bit
        - Stage: sft
        
        Provide parameter names, types, and defaults.
        """
        
        result = agent.query(query)
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention LoRA-related parameters
        assert "lora" in result.lower()

    def test_method_comparison_query(self, agent):
        """Test comparing different fine-tuning methods"""
        result = agent.query("What are the key differences between LoRA and Freeze methods?")
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_complex_configuration_query(self, agent):
        """Test querying complex configuration"""
        query = """
        Get all relevant parameters for:
        - Method: lora
        - Quantization: 4bit
        - Optimizer: standard
        - Distributed: single
        
        Group by importance level.
        """
        
        result = agent.query(query)
        
        assert result is not None
        assert isinstance(result, str)
        # Should return substantial parameter information
        assert len(result) > 100


def test_agent_standalone():
    """Standalone test that can be run directly"""
    try:
        agent = Agent()
        result = agent.query("List the most important parameters for LoRA fine-tuning")
        print("\n" + "="*80)
        print("Query Result:")
        print("="*80)
        print(result)
        print("="*80)
        assert result is not None
    except Exception as e:
        pytest.skip(f"MCP agent not available: {e}")


if __name__ == "__main__":
    # Quick manual test
    print("Testing LLaMA Factory MCP Agent...")
    print("\n1. Testing agent initialization...")
    try:
        agent = Agent()
        print("✓ Agent initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        print("\nMake sure MCP server is running:")
        print("  python -m rdagent.components.agent.mcp.servers.llama_factory_server")
        exit(1)
    
    print("\n2. Testing simple query...")
    result = agent.query("What is lora_rank?")
    print(f"✓ Query successful, result length: {len(result)} chars")
    print(f"\nResult preview:\n{result[:200]}...")
    
    print("\n3. Testing decision-based query...")
    result = agent.query("Get parameters for: method=lora, quantization=4bit")
    print(f"✓ Query successful, result length: {len(result)} chars")
    
    print("\n" + "="*80)
    print("All manual tests passed!")
    print("="*80)

