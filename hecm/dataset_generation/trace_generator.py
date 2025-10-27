import cursor_agent_tools.factory as factory
from cursor_agent_tools.openai_agent import OpenAIAgent
import asyncio

# -------------------------------
# Runtime patch for local vLLM
# -------------------------------
def patched_create_agent(*args, **kwargs):
    # remove model from kwargs to avoid double passing
    model = kwargs.pop("model", None) or (args[0] if args else None)
    
    # if model starts with 'archit11/', treat it as local vLLM
    if model and model.startswith("archit11/"):
        print(f"[patched factory] Detected local vLLM model: {model}")
        return OpenAIAgent(
            model=model,
            api_key="EMPTY",              # not used for local vLLM
            base_url="http://localhost:8005/v1",  # local vLLM endpoint
            api_type="custom",            # ensures it uses base_url only
            temperature=kwargs.get("temperature", 0.0),
            timeout=kwargs.get("timeout", 180),
            permission_callback=kwargs.get("permission_callback"),
            permission_options=kwargs.get("permissions"),
            default_tool_timeout=kwargs.get("default_tool_timeout", 300),
            **kwargs,
        )
    
    # fallback to original factory
    return factory._orig_create_agent(*args, **kwargs)

# Patch the factory at runtime
factory._orig_create_agent = factory.create_agent
factory.create_agent = patched_create_agent

# -------------------------------
# Import after patch to pick up patched function
# -------------------------------
from cursor_agent_tools.factory import create_agent

# -------------------------------
# Async test function
# -------------------------------
async def test_local_vllm():
    ollama_agent = create_agent(model="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    response = await ollama_agent.chat("Hello, can you summarize caching strategies in Python?")
    print(response)

# Run async function
if __name__ == "__main__":
    asyncio.run(test_local_vllm())
