import os
import subprocess
import asyncio
import aiohttp

async def main():
    print("Testing Ollama connection and models...")
    
    # Check if Ollama is running
    try:
        process = await asyncio.create_subprocess_exec(
            "pgrep", "ollama",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        
        if stdout:
            print("✅ Ollama process is running")
        else:
            print("❌ Ollama process is not running!")
            print("Attempting to start Ollama...")
            
            try:
                start_process = await asyncio.create_subprocess_exec(
                    "ollama", "serve",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True
                )
                # Wait a moment for Ollama to start
                await asyncio.sleep(3)
                print("Started Ollama server")
            except Exception as e:
                print(f"Failed to start Ollama: {e}")
    except Exception as e:
        print(f"Error checking Ollama process: {e}")
    
    # Check available models
    try:
        process = await asyncio.create_subprocess_exec(
            "ollama", "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print("\nAvailable Ollama models:")
            print(stdout.decode())
        else:
            print(f"Error listing models: {stderr.decode()}")
    except Exception as e:
        print(f"Error listing models: {e}")
    
    # Try to connect to Ollama API
    try:
        ollama_api_base = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434")
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ollama_api_base}/api/tags") as response:
                if response.status == 200:
                    print("✅ Successfully connected to Ollama API")
                    models_data = await response.json()
                    if models_data.get("models"):
                        print(f"API reports {len(models_data['models'])} available models")
                else:
                    print(f"❌ Ollama API error: {response.status}")
                    error_text = await response.text()
                    print(f"Error message: {error_text}")
    except Exception as e:
        print(f"❌ Failed to connect to Ollama API: {e}")
    
    # Test CellBot's check_ollama_status method directly
    try:
        print("\nTesting CellBot's check_ollama_status method...")
        from agent import MinimalAIAgent
        agent = MinimalAIAgent()
        status = await agent.check_ollama_status()
        print(f"Result: {status}")
    except Exception as e:
        print(f"❌ Error testing check_ollama_status: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 