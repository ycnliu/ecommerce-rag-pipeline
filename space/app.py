#!/usr/bin/env python3
"""
Lightweight HuggingFace Spaces demo for E-commerce RAG Pipeline.
This is a demo-only version with minimal dependencies.
"""
import csv
import json
import os
from typing import Dict, List

import gradio as gr


# Load demo products
def load_demo_products() -> List[Dict]:
    """Load demo products from CSV."""
    products = []

    try:
        with open("demo_products.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                products.append(row)
    except FileNotFoundError:
        # Fallback to hardcoded demo products
        products = [
            {
                "name": "Sony WH-CH720N Wireless Headphones",
                "price": "89.99",
                "category": "Electronics",
                "description": "Active Noise Cancelling, 35-hour battery",
            },
            {
                "name": "JBL Tune 510BT Wireless Headphones",
                "price": "39.99",
                "category": "Audio",
                "description": "Wireless Bluetooth 5.0, Pure Bass sound",
            },
        ]

    return products


# Simple search function
def search_products(query: str, products: List[Dict]) -> List[Dict]:
    """Simple keyword-based product search."""
    if not query:
        return products[:5]

    query_lower = query.lower()
    results = []

    for product in products:
        text = (
            f"{product.get('name', '')} "
            f"{product.get('description', '')} "
            f"{product.get('category', '')}"
        ).lower()

        if any(word in text for word in query_lower.split()):
            results.append(product)
            if len(results) >= 5:
                break

    return results[:5] if results else products[:5]


# Generate AI-like response
def generate_response(query: str, results: List[Dict]) -> str:
    """Generate template-based recommendation."""
    if not results:
        return "No products found matching your query."

    response = f"Based on your search for '{query}', here are my recommendations:\n\n"

    for i, product in enumerate(results[:3], 1):
        name = product.get("name", "Unknown")
        price = product.get("price", "N/A")
        desc = product.get("description", "No description")

        response += f"{i}. **{name}**\n"
        response += f"   - Price: ${price}\n"
        response += f"   - {desc}\n\n"

    response += "\nConsider comparing features, reading reviews, and choosing based on your budget."

    return response


# Load products
PRODUCTS = load_demo_products()


# Process query
def process_query(query: str) -> tuple:
    """Process user query and return results."""
    if not query or query.strip() == "":
        return "Please enter a search query.", "Enter a query to get recommendations."

    # Search products
    results = search_products(query, PRODUCTS)

    # Format search results
    if results:
        search_md = "## Search Results\n\n"
        for i, product in enumerate(results, 1):
            search_md += f"### {i}. {product.get('name', 'Unknown')}\n"
            search_md += f"**Price:** ${product.get('price', 'N/A')}\n\n"
            search_md += f"**Category:** {product.get('category', 'N/A')}\n\n"
            search_md += f"{product.get('description', 'No description')}\n\n"
            search_md += "---\n\n"
    else:
        search_md = "No products found."

    # Generate AI response
    ai_response = generate_response(query, results)

    return search_md, ai_response


# Create Gradio interface
def create_interface():
    """Create the Gradio demo interface."""

    with gr.Blocks(title="E-commerce RAG Pipeline Demo") as demo:

        gr.HTML(
            """
        <div style="text-align: center; margin-bottom: 2em;">
            <h1>E-commerce RAG Pipeline</h1>
            <p>AI-powered product search and recommendations (Demo Version)</p>
            <p style="font-size: 0.9em; color: #666;">
                Built with CI/CD | Deployed via GitHub Actions |
                <a href="https://github.com/ycnliu/ecommerce-rag-pipeline">GitHub</a>
            </p>
        </div>
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., 'wireless headphones under $100'",
                    lines=2,
                )

                search_btn = gr.Button("Search Products", variant="primary", size="lg")

                gr.HTML(
                    """
                <div style="margin-top: 1em; padding: 1em; background: #f5f5f5; border-radius: 8px;">
                    <h4>Example Queries:</h4>
                    <ul style="margin: 0.5em 0;">
                        <li>wireless bluetooth headphones</li>
                        <li>educational toys for kids</li>
                        <li>electronics under $50</li>
                    </ul>
                </div>
                """
                )

        with gr.Row():
            with gr.Column(scale=1):
                search_output = gr.Markdown(
                    label="Product Results", value="Search results will appear here..."
                )

            with gr.Column(scale=1):
                ai_output = gr.Markdown(
                    label="AI Recommendations",
                    value="AI recommendations will appear here...",
                )

        gr.HTML(
            """
        <div style="margin-top: 2em; padding: 1.5em; background: #f9f9f9; border-radius: 10px;">
            <h3>About This Demo</h3>
            <p><strong>Technology:</strong> Retrieval-Augmented Generation (RAG) pipeline</p>
            <p><strong>Architecture:</strong> CLIP embeddings + FAISS vector search + LLM generation</p>
            <p><strong>Deployment:</strong> Automated via GitHub Actions with quality gates</p>
            <p><strong>Full Source:</strong>
                <a href="https://github.com/ycnliu/ecommerce-rag-pipeline">github.com/ycnliu/ecommerce-rag-pipeline</a>
            </p>
            <p style="font-size: 0.9em; color: #666; margin-top: 1em;">
                Note: This is a lightweight demo. Full version includes real CLIP embeddings,
                FAISS indexing, and LLM integration.
            </p>
        </div>
        """
        )

        # Event handlers
        search_btn.click(
            fn=process_query, inputs=[query_input], outputs=[search_output, ai_output]
        )

        query_input.submit(
            fn=process_query, inputs=[query_input], outputs=[search_output, ai_output]
        )

    return demo


# Launch
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
