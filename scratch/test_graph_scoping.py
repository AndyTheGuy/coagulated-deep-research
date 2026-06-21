import asyncio
import os
import structlog
from core.graph import compile_graph

# Configure structlog to output to stdout for this scratch run
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

async def test_scoping():
    os.environ["MOCK_LLM"] = "true"
    initial_state = {
        "user_query": "Autonomous Python Standard Library Agent Architectures",
        "topic": "Python StdLib",
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
    print("Graph compiled successfully. Invoking graph...")
    
    try:
        final_state = await app.ainvoke(initial_state)
        print("\n--- FINAL STATE ---")
        print("topic:", final_state.get("topic"))
        print("clarification_needed:", final_state.get("clarification_needed"))
        print("research_brief exists:", final_state.get("research_brief") is not None)
        print("errors:", final_state.get("errors"))
        print("sub_questions count:", len(final_state.get("sub_questions_state", [])))
    except Exception as e:
        print("ERROR running graph:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_scoping())
