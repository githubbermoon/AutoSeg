
import os
import sys
from huggingface_hub import create_repo, whoami, HfApi

def create_space():
    print("🚀 AutoSeg Space Creator")
    print("-----------------------")
    
    # 1. Check Login
    try:
        user_info = whoami()
        username = user_info['name']
        print(f"✅ Logged in as: {username}")
    except Exception as e:
        print("❌ Not logged in to Hugging Face CLI.")
        print("   Please run: huggingface-cli login")
        print("   (Or set HF_TOKEN environment variable)")
        return

    # 2. Ask for Space Name or use CLI arg
    default_name = "AutoSeg-Demo"
    if len(sys.argv) > 1:
        space_name = sys.argv[1]
        print(f"👉 Using CLI provided name: {space_name}")
    else:
        # Default for non-interactive run if input fails (e.g. invalid file descriptor)
        if not sys.stdin.isatty():
             print(f"⚠️  Non-interactive mode detected. Using default: {default_name}")
             space_name = default_name
        else:
             space_name = input(f"Enter Space Name [default: {default_name}]: ").strip() or default_name
    
    repo_id = f"{username}/{space_name}"
    
    print(f"\nTarget Repo: {repo_id}")
    
    # 3. Create Space
    try:
        url = create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False, # Make public by default? Or ask?
            exist_ok=True
        )
        print(f"✅ Space created successfully!")
        print(f"🔗 URL: {url}")
        
        # 4. Upload Files (Initial Push)
        print("\nUploading files...")
        api = HfApi()
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[".git", ".env", "__pycache__", "wandb", ".vscode", "tests"]
        )
        print("✅ Files uploaded!")
        print(f"👉 Go to {url} to see your app running.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    create_space()
