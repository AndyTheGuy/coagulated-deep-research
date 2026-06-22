import asyncio
import sys
import os
import yaml
import traceback
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage

# Try to reconfigure stdout to UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Define colors for terminal output (or fallback to plain if not supported)
GREEN = "\033[92m" if sys.stdout.isatty() else ""
RED = "\033[91m" if sys.stdout.isatty() else ""
YELLOW = "\033[93m" if sys.stdout.isatty() else ""
CYAN = "\033[96m" if sys.stdout.isatty() else ""
RESET = "\033[0m" if sys.stdout.isatty() else ""

async def test_model(model_id: str, project_id: str, location: str, model_name: str) -> dict:
    """Test a single model on Vertex AI and return result dict."""
    print(f"{CYAN}Testing '{model_name}' ({model_id}) on project '{project_id}' in region '{location}'...{RESET}")
    result = {
        "model_name": model_name,
        "model_id": model_id,
        "project": project_id,
        "location": location,
        "success": False,
        "response": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "error": None
    }
    
    try:
        model = ChatVertexAI(
            model=model_id,
            project=project_id,
            location=location,
        )
        messages = [HumanMessage(content="Hello, answer with exactly one word: Success.")]
        
        # Increase timeout slightly to 20 seconds and run
        response = await asyncio.wait_for(model.ainvoke(messages), timeout=20.0)
        
        # Extract text content cleanly
        content = response.content
        if isinstance(content, list):
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
            
        result["success"] = True
        result["response"] = response_text.strip()
        
        # Token usage metadata extraction
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            result["input_tokens"] = response.usage_metadata.get("input_tokens", 0)
            result["output_tokens"] = response.usage_metadata.get("output_tokens", 0)
        elif "token_usage" in getattr(response, "response_metadata", {}):
            usage = response.response_metadata["token_usage"]
            result["input_tokens"] = usage.get("prompt_tokens", 0)
            result["output_tokens"] = usage.get("completion_tokens", 0)
            
        print(f"{GREEN}[OK] {model_name} responded: '{result['response']}' (Tokens: In={result['input_tokens']}, Out={result['output_tokens']}){RESET}")
        
    except Exception as e:
        result["success"] = False
        # Capture class name and details
        err_type = type(e).__name__
        err_msg = str(e) or "No error message provided"
        result["error"] = f"{err_type}: {err_msg}"
        print(f"{RED}[FAIL] {model_name} failed: {result['error']}{RESET}")
        
    return result

def load_gemini_models_from_yaml(config_path: str) -> list:
    """Parse Model Garden YAML config and extract Vertex AI Gemini models."""
    print(f"Loading configurations from: {config_path}")
    if not os.path.exists(config_path):
        print(f"{RED}Config file not found: {config_path}{RESET}")
        return []
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    model_list = config.get("model_list", [])
    gemini_models = []
    
    for item in model_list:
        model_name = item.get("model_name", "")
        litellm_params = item.get("litellm_params", {})
        model_id = litellm_params.get("model", "")
        
        # Check if it represents a native Vertex AI Gemini model
        if model_id.startswith("vertex_ai/gemini-"):
            # Extract actual Vertex AI model identifier (everything after 'vertex_ai/')
            actual_model_id = model_id[len("vertex_ai/"):]
            project_id = litellm_params.get("vertex_project", "agenticuse")
            location = litellm_params.get("vertex_location", "global")
            
            gemini_models.append({
                "model_name": model_name,
                "model_id": actual_model_id,
                "project": project_id,
                "location": location
            })
            
    print(f"Found {len(gemini_models)} Gemini Vertex AI models in config.")
    return gemini_models

async def main():
    config_path = r"C:\Users\beste\Documents\claude_code\configuring-claude\config.yaml"
    gemini_models = load_gemini_models_from_yaml(config_path)
    
    if not gemini_models:
        print(f"{RED}No models found to test.{RESET}")
        sys.exit(1)
        
    print("\nStarting Sequential Automated Verification Loop for all Gemini Model Garden models...")
    results = []
    for model_info in gemini_models:
        res = await test_model(
            model_id=model_info["model_id"],
            project_id=model_info["project"],
            location=model_info["location"],
            model_name=model_info["model_name"]
        )
        results.append(res)
        # Yield to event loop to be safe
        await asyncio.sleep(0.5)
    
    print("\n" + "="*80)
    print("MODEL GARDEN GEMINI VERIFICATION REPORT")
    print("="*80)
    print(f"{'Model Garden Name':<35} | {'Vertex Model ID':<30} | {'Status':<10}")
    print("-" * 80)
    
    success_count = 0
    for r in results:
        status_str = "WORKING" if r["success"] else "FAILED"
        if r["success"]:
            success_count += 1
        print(f"{r['model_name']:<35} | {r['model_id']:<30} | {status_str}")
        if not r["success"]:
            print(f"  |- Error: {r['error']}")
            
    print("="*80)
    print(f"Passed: {success_count}/{len(results)} models.")
    print("="*80)
    
    if success_count == len(results):
        print("All configured Gemini models are fully functional!")
        sys.exit(0)
    else:
        print("Some models are unavailable (possibly due to region, preview status, or quota limits).")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
