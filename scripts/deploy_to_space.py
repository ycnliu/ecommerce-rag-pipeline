#!/usr/bin/env python3
"""
Deploy to HuggingFace Spaces.
"""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def deploy_to_space(space_id: str, source_dir: str, commit_message: str):
    """Deploy files to HuggingFace Space."""

    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    api = HfApi(token=hf_token)

    print(f"Deploying to Space: {space_id}")
    print(f"Source directory: {source_dir}")

    # Create Space if it doesn't exist
    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            token=hf_token
        )
        print(f"Space ready: {space_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Upload files
    print("Uploading files...")

    upload_folder(
        folder_path=source_dir,
        repo_id=space_id,
        repo_type="space",
        commit_message=commit_message,
        token=hf_token
    )

    print(f"Deployment complete!")
    print(f"Space URL: https://huggingface.co/spaces/{space_id}")


def main():
    parser = argparse.ArgumentParser(description='Deploy to HuggingFace Spaces')
    parser.add_argument('--space-id', required=True, help='Space ID (username/space-name)')
    parser.add_argument('--source-dir', required=True, help='Source directory to deploy')
    parser.add_argument('--commit-message', required=True, help='Commit message')

    args = parser.parse_args()

    deploy_to_space(args.space_id, args.source_dir, args.commit_message)


if __name__ == '__main__':
    main()
