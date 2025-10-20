"""
Direct test of LLaMA Factory MCP server core functions
No MCP protocol, no LLM, just pure function calls

Usage:
    python rdagent/components/agent/mcp/test_alone.py
"""

import json
import sys
from pathlib import Path
import dotenv
dotenv.load_dotenv('.env')

# Add LLaMA Factory to path
LLAMA_FACTORY_PATH = Path(__file__).parent.parent.parent.parent.parent / "git_ignore_folder" / "LLaMA-Factory" / "src"
if LLAMA_FACTORY_PATH.exists():
    sys.path.insert(0, str(LLAMA_FACTORY_PATH))
else:
    print(f"❌ LLaMA Factory not found at: {LLAMA_FACTORY_PATH}")
    sys.exit(1)

from rdagent.components.agent.mcp.servers.llama_factory_server import SimpleLLaMAFactoryIntrospector


def test_capabilities():
    """Test get_capabilities function"""
    print("\n" + "="*80)
    print("TEST 1: Get Capabilities")
    print("="*80)
    
    introspector = SimpleLLaMAFactoryIntrospector()
    capabilities = introspector.get_capabilities()
    
    print(f"\n✅ Methods: {capabilities['methods']}")
    print(f"✅ Stages: {capabilities['stages']}")
    print(f"✅ Optimizers: {len(capabilities['optimizers'])} options")
    for optimizer in capabilities['optimizers']:
        print(f"   - {optimizer}")
    print(f"✅ Schedulers: {len(capabilities['schedulers'])} options")
    for scheduler in capabilities['schedulers']:
        print(f"   - {scheduler}")
    
    return capabilities


def test_parameter_schema_category(category: str):
    """Test get_parameter_schema for specific category"""
    print(f"\n{'='*80}")
    print(f"TEST: Get Parameter Schema - {category.upper()}")
    print("="*80)
    
    introspector = SimpleLLaMAFactoryIntrospector()
    schema = introspector.get_parameter_schema(category)
    
    for cat_name, cat_schema in schema.items():
        print(f"\n📦 {cat_name.upper()}")
        print(f"   Class: {cat_schema['class_name']}")
        print(f"   Total params: {cat_schema['total_params']}")
        print(f"   Inheritance: {' -> '.join(cat_schema['inheritance_chain'])}")
        
        # Show first 5 parameters
        params = list(cat_schema['parameters'].items())[:5]
        print(f"\n   Sample parameters:")
        for param_name, param_info in params:
            default = param_info['default']
            param_type = param_info['type']
            desc = param_info['description'][:50] if param_info['description'] else "No description"
            print(f"   - {param_name}: {param_type} = {default}")
            print(f"     {desc}...")
    
    return schema


def test_specific_parameters():
    """Test looking up specific important parameters"""
    print("\n" + "="*80)
    print("TEST: Lookup Specific Parameters")
    print("="*80)
    
    introspector = SimpleLLaMAFactoryIntrospector()
    
    # Get finetuning parameters
    schema = introspector.get_parameter_schema("finetuning")
    finetuning_params = schema["finetuning"]["parameters"]
    
    # Check key LoRA parameters
    key_params = ["lora_rank", "lora_alpha", "lora_dropout", "finetuning_type"]
    
    print("\n🔍 Key LoRA Parameters:")
    for param in key_params:
        if param in finetuning_params:
            info = finetuning_params[param]
            print(f"\n✅ {param}:")
            print(f"   Type: {info['type']}")
            print(f"   Default: {info['default']}")
            print(f"   Description: {info['description']}")
        else:
            print(f"\n❌ {param}: Not found")


def test_all_categories_stats():
    """Get statistics for all categories"""
    print("\n" + "="*80)
    print("TEST: All Categories Statistics")
    print("="*80)
    
    introspector = SimpleLLaMAFactoryIntrospector()
    all_schema = introspector.get_parameter_schema("all")
    
    total = 0
    print("\n📊 Parameter Count by Category:")
    for cat_name, cat_schema in all_schema.items():
        count = cat_schema['total_params']
        total += count
        print(f"   {cat_name:15s}: {count:3d} parameters")
    
    print(f"\n   {'TOTAL':15s}: {total:3d} parameters")
    
    return all_schema


def test_json_export():
    """Test exporting to JSON (useful for debugging)"""
    print("\n" + "="*80)
    print("TEST: JSON Export")
    print("="*80)
    
    introspector = SimpleLLaMAFactoryIntrospector()
    
    # Get training parameters only (smaller output)
    schema = introspector.get_parameter_schema("training")
    
    # Save to file
    output_file = "/tmp/llama_factory_training_params.json"
    with open(output_file, "w") as f:
        json.dump(schema, f, indent=2)
    
    print(f"\n✅ Exported to: {output_file}")
    print(f"   File size: {Path(output_file).stat().st_size} bytes")
    
    # Show first 100 chars
    with open(output_file) as f:
        preview = f.read(200)
    print(f"\n   Preview:\n{preview}...")


def main():
    """Run all tests"""
    print("\n" + "#"*80)
    print("# LLaMA Factory MCP Server - Direct Function Test")
    print("#"*80)
    
    try:
        # Test 1: Capabilities
        capabilities = test_capabilities()
        
        # Test 2: Individual categories
        test_parameter_schema_category("finetuning")
        test_parameter_schema_category("training")
        
        # Test 3: Specific parameters
        test_specific_parameters()
        
        # Test 4: Statistics
        all_schema = test_all_categories_stats()
        
        # Test 5: JSON export
        test_json_export()
        
        # Summary
        print("\n" + "#"*80)
        print("# Test Summary")
        print("#"*80)
        print(f"\n✅ All tests passed!")
        print(f"✅ Methods: {len(capabilities['methods'])}")
        print(f"✅ Stages: {len(capabilities['stages'])}")
        print(f"✅ Total parameters extracted: {sum(s['total_params'] for s in all_schema.values())}")
        print(f"\n🎉 Core functionality verified without MCP/LLM overhead!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

