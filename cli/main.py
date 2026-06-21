import asyncio
import sys
from core.graph import compile_graph

async def main() -> None:
    """Run the interactive CLI loop for deep research scoping."""
    print("==================================================")
    print("Ultimate Deep Researcher - CLI Scoping Tool")
    print("==================================================")
    
    query = input("\nEnter your research query: ").strip()
    if not query:
        print("Error: query cannot be empty.")
        sys.exit(1)
        
    app = compile_graph()
    
    state = {
        "user_query": query
    }
    
    print("\nRunning scoping agent...")
    result = await app.ainvoke(state)
    
    # Handle clarification loop
    if result.get("clarification_needed"):
        question = result.get("clarification_question")
        print(f"\n[Clarification Needed] {question}")
        response = input("Your response: ").strip()
        
        # Re-run graph with the response
        state_update = {
            "user_query": query,
            "clarification_question": question,
            "clarification_response": response,
            "clarification_needed": False
        }
        print("\nRe-running scoping agent with clarification...")
        result = await app.ainvoke(state_update)
        
    # Output the completed research brief
    brief = result.get("research_brief")
    if brief:
        print("\n==================================================")
        print("Generated Research Brief")
        print("==================================================")
        print(brief.model_dump_json(indent=2))
        print("==================================================")
    else:
        print("\nFailed to generate research brief.")
        if result.get("errors"):
            print(f"Errors: {result.get('errors')}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
