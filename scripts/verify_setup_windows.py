import os
import sys
import subprocess
from pathlib import Path

def main():
    print("OpenJarvis Windows Setup Verification")
    print("======================================")


    # 2. Config loading
    print("\n[2] Checking Configuration...")
    try:
        # Add src to sys.path
        src_path = str(Path(__file__).parent.parent / "src")
        if src_path not in sys.path:
            sys.path.append(src_path)
            
        from openjarvis.core.config import load_config
        repo_config = Path(__file__).parent.parent / "configs" / "openjarvis" / "config.toml"
        if repo_config.exists():
            config = load_config(path=repo_config)
            print(f"  [OK] Loaded repo config: {repo_config}")
        else:
            config = load_config()
            print("  [OK] Loaded default config (~/.openjarvis/config.toml)")
        
        # 1. Environment variables (checked after load_config to see .env)
        print("\n[1] Checking Environment Variables...")
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            print(f"  [OK] GROQ_API_KEY is set (starts with {groq_api_key[:4]}...)")
        else:
            print("  [ERROR] GROQ_API_KEY is NOT set in environment or .env")
            print("     Action: Add GROQ_API_KEY=your_key to a .env file in the root directory.")
        
        # 3. Llama.cpp checks (if configured)
        engine_default = config.engine.default
        print(f"  Info: Default engine is '{engine_default}'")

        if engine_default == "llama_cpp" or "llama_cpp" in config.engine.channel_overrides.values():
            print("\n[3] Checking llama_cpp requirements...")
            lcpp = config.engine.llama_cpp
            
            # Model path
            if lcpp.model_path:
                model_path = Path(lcpp.model_path).expanduser()
                if model_path.exists():
                    print(f"  [OK] Model file exists: {model_path}")
                else:
                    print(f"  [ERROR] Model file NOT found: {model_path}")
            else:
                print("  [ERROR] llama_cpp model_path is not configured")

            # Binary path
            binary = lcpp.binary_path or "llama-server"
            if sys.platform == "win32" and not binary.lower().endswith(".exe"):
                import shutil
                binary_resolved = shutil.which(binary) or shutil.which(f"{binary}.exe")
            else:
                import shutil
                binary_resolved = shutil.which(binary)

            if binary_resolved:
                print(f"  [OK] llama-server binary found at: {binary_resolved}")
            else:
                print(f"  [ERROR] llama-server binary NOT found in PATH or config")
                print(f"     Action: Ensure llama.cpp binaries are installed and 'llama-server.exe' is in PATH")
        
        # 4. Engine Health Check
        print("\n[4] Performing Engine Health Check...")
        from openjarvis.engine import get_engine
        
        # Test default engine
        res = get_engine(config)
        if res:
            engine_name, engine_inst = res
            print(f"  Info: Testing engine '{engine_name}'...")
            if engine_inst.health():
                print(f"  [OK] Engine '{engine_name}' is HEALTHY")
            else:
                print(f"  [ERROR] Engine '{engine_name}' health check FAILED")
                if engine_name == "llama_cpp":
                    print("     Action: Ensure llama-server can start (check model path and port).")
                elif engine_name == "groq":
                    print("     Action: Check your GROQ_API_KEY and internet connection.")
        else:
            print("  [ERROR] Could not resolve default engine")

    except Exception as exc:
        print(f"  [ERROR] Error during verification: {exc}")
        import traceback
        traceback.print_exc()

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
