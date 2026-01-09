#!/usr/bin/env python3
"""
Verify HuggingFace Space is running correctly.
"""
import argparse
import time
import requests


def verify_space(space_url: str, max_retries: int = 5):
    """Verify Space is accessible and responding."""

    print(f"Verifying Space: {space_url}")

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")

            response = requests.get(space_url, timeout=30)

            if response.status_code == 200:
                print("Space is accessible!")

                # Check for expected content
                if "E-commerce RAG Pipeline" in response.text:
                    print("Space content verified!")
                    return True
                else:
                    print("Warning: Expected content not found")

            else:
                print(f"HTTP {response.status_code}")

        except requests.RequestException as e:
            print(f"Error: {e}")

        if attempt < max_retries - 1:
            wait_time = 10 * (attempt + 1)
            print(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

    raise RuntimeError(f"Space verification failed after {max_retries} attempts")


def main():
    parser = argparse.ArgumentParser(description="Verify HuggingFace Space")
    parser.add_argument("--space-url", required=True, help="Space URL")
    parser.add_argument(
        "--max-retries", type=int, default=5, help="Maximum retry attempts"
    )

    args = parser.parse_args()

    verify_space(args.space_url, args.max_retries)
    print("Verification successful!")


if __name__ == "__main__":
    main()
