#!/usr/bin/env python3
"""
Prepare demo dataset for HuggingFace Spaces deployment.
Extracts a subset of products with minimal required fields.
"""
import argparse
import pandas as pd
from pathlib import Path


def prepare_demo_data(input_path: str, output_path: str, limit: int = 100):
    """Extract and prepare demo dataset."""

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Total products: {len(df)}")

    # Select relevant columns
    columns_to_keep = [
        "Product Name",
        "Selling Price",
        "Category",
        "About Product",
        "Product Description",
    ]

    # Filter columns that exist
    available_columns = [col for col in columns_to_keep if col in df.columns]

    if not available_columns:
        print("Warning: No standard columns found, using first 5 columns")
        available_columns = df.columns[:5].tolist()

    # Take subset
    demo_df = df[available_columns].head(limit).copy()

    # Rename columns to standard names
    column_mapping = {
        "Product Name": "name",
        "Selling Price": "price",
        "Category": "category",
        "About Product": "description",
        "Product Description": "description_long",
    }

    demo_df.rename(columns=column_mapping, inplace=True)

    # Clean data
    demo_df = demo_df.fillna("Not available")

    # Combine descriptions if both exist
    if "description" in demo_df.columns and "description_long" in demo_df.columns:
        demo_df["description"] = demo_df.apply(
            lambda row: (
                row["description"]
                if pd.notna(row["description"])
                and row["description"] != "Not available"
                else (
                    row["description_long"]
                    if pd.notna(row["description_long"])
                    else "No description"
                )
            ),
            axis=1,
        )
        demo_df = demo_df.drop("description_long", axis=1)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    demo_df.to_csv(output_path, index=False)

    print(f"Demo dataset saved to {output_path}")
    print(f"Products: {len(demo_df)}")
    print(f"Columns: {list(demo_df.columns)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare demo dataset for HF Spaces")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument(
        "--limit", type=int, default=100, help="Number of products to include"
    )

    args = parser.parse_args()

    prepare_demo_data(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
