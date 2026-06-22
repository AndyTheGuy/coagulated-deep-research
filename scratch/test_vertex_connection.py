import asyncio
import sys
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage

async def test_model(model_name: str, project_id: str, location: str):
    print(f"\n--- Testing model: '{model_name}' on project '{project_id}' in region '{location}' ---")
    try:
        model = ChatVertexAI(
            model=model_name,
            project=project_id,
            location=location,
        )
        print("Initializing connection...")
        messages = [HumanMessage(content="Hello, answer with exactly one word: Success.")]
        response = await model.ainvoke(messages)
        
        # Safely extract response text content
        content = response.content
        if isinstance(content, list):
            # Parse block list
            text_parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text_parts.append(block["text"])
                elif hasattr(block, "text"):
                    text_parts.append(block.text)
                else:
                    text_parts.append(str(block))
            response_text = " ".join(text_parts)
        else:
            response_text = str(content)
            
        print(f"Response: '{response_text.strip()}'")
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            print(f"Token metadata: {response.usage_metadata}")
            
        print(f"[OK] Model '{model_name}' is active and working!")
        return True
    except Exception as e:
        print(f"[FAIL] Model '{model_name}' failed to respond!")
        print(f"Error Details: {str(e)}")
        return False

async def main():
    project_id = "agenticuse"
    location = "global"
    
    # Models to test from the approved model garden
    models_to_test = [
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro"
    ]
    
    results = {}
    for model_name in models_to_test:
        success = await test_model(model_name, project_id, location)
        results[model_name] = success
        
    print("\n" + "="*50)
    print("VERTEX AI MODEL CONNECTION REPORT:")
    print("="*50)
    for model_name, success in results.items():
        status = "WORKING" if success else "FAILED"
        print(f"* {model_name:25} : {status}")
    print("="*50)
    
    if all(results.values()):
        print("All approved model connections passed successfully!")
        sys.exit(0)
    else:
        print("Some model connections failed. Please check project configuration/quotas.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
