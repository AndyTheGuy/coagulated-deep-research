import asyncio
from duckduckgo_search import DDGS

async def main():
    query = '"Define AI safety benchmarks and their role in evaluation"'
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            print("Quoted results (auto):", list(results) if results else [])
    except Exception as e:
        print("Quoted failed:", e)

    # Test sanitized query
    sanitized = query.strip('"')
    try:
        with DDGS() as ddgs:
            results = ddgs.text(sanitized, max_results=3)
            print("Sanitized results (auto):", list(results) if results else [])
    except Exception as e:
        print("Sanitized failed:", e)

if __name__ == "__main__":
    asyncio.run(main())
