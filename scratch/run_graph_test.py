import asyncio
import os
import sys

# Ensure PYTHONPATH includes the current directory
sys.path.insert(0, os.getcwd())

# Set mock LLM mode to true
os.environ["MOCK_LLM"] = "true"

from core.graph import compile_graph

async def main():
    initial_state = {
        "user_query": "Autonomous Python Standard Library Agent Architectures",
        "topic": "Autonomous Python Standard Library Agent Architectures",
        "sub_questions_state": [],
        "search_results": [],
        "verified_sources": [],
        "claims": [],
        "errors": [],
        "logs": [],
        "token_usage": {
            "vertex_ai": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
            "freellmapi": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
            "failovers": 0
        }
    }
    
    app = compile_graph()
    print("Graph compiled successfully. Running stream...")
    
    try:
        async for chunk in app.astream(initial_state, stream_mode="updates"):
            print("--- CHUNK RECEIVED ---")
            for node_name, node_update in chunk.items():
                print(f"Node: {node_name}")
                if node_update is None:
                    print("  Update: None")
                    continue
                if not isinstance(node_update, dict):
                    print(f"  Update: {node_update}")
                    continue
                print(f"Update Keys: {list(node_update.keys())}")
                if "clarification_needed" in node_update:
                    print(f"  clarification_needed: {node_update['clarification_needed']}")
                if "research_brief" in node_update:
                    print(f"  research_brief: {node_update['research_brief']}")
                if "sub_questions_state" in node_update:
                    print(f"  sub_questions_state: {len(node_update['sub_questions_state'])} items")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
