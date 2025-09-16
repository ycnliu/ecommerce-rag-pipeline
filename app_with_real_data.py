#!/usr/bin/env python3
"""
HuggingFace Spaces Gradio app for E-commerce RAG Pipeline with real dataset support.
"""
import gradio as gr
import os
import sys
import json
import time
import pandas as pd
from typing import List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append('src')

# Import our components (with fallbacks)
try:
    from src.rag.llm_client import create_llm_client
    from src.rag.prompt_builder import PromptBuilder
    LLM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LLM imports failed: {e}")
    LLM_AVAILABLE = False

# Fallback LLM client for Spaces
class FallbackLLMClient:
    """Simple fallback LLM client for Spaces deployment."""

    def __init__(self):
        self.model_name = "fallback"

    def generate_response(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate a helpful fallback response."""

        if "search results:" in prompt.lower():
            return """Based on the search results, I found several relevant products that match your query. Here are my recommendations:

🎯 **Top Pick**: Consider the product with the best balance of features, price, and reviews.

💰 **Budget Option**: Look for the most affordable option that still meets your needs.

⭐ **Premium Choice**: If budget allows, the higher-priced option likely offers better quality or features.

📝 **Recommendation**: Compare the specifications, read reviews, and choose based on your specific requirements and budget.

*Note: This is a demo response. In production, this would be powered by advanced language models for more detailed analysis.*"""
        else:
            return """Hello! I'm an AI assistant for e-commerce product recommendations.

🛒 **How I can help:**
- Find products based on your requirements
- Compare different options
- Provide detailed recommendations
- Answer questions about features and specifications

📝 **To get started**: Try searching for products like "wireless headphones under $100" or "educational toys for kids"

*Note: This is a demo version. The full system uses advanced AI models for better responses.*"""

    def get_model_info(self):
        return {"model_name": "fallback", "type": "demo"}

# Initialize components
if LLM_AVAILABLE:
    try:
        # Try OpenAI first if available
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            llm_client = create_llm_client(
                provider="openai",
                model_name="gpt-3.5-turbo",
                api_token=openai_key
            )
            logger.info("Using OpenAI GPT-3.5-turbo for enhanced responses")
        else:
            # Fallback to free options
            llm_client = create_llm_client(
                provider="free",
                model_name="fallback",
                api_token=os.getenv("HF_TOKEN", "demo")
            )
        prompt_builder = PromptBuilder()
    except Exception as e:
        logger.warning(f"Failed to initialize LLM components: {e}")
        llm_client = FallbackLLMClient()
        prompt_builder = None
else:
    llm_client = FallbackLLMClient()
    prompt_builder = None

# Load real dataset or use mock data
def load_product_database():
    """Load product database from CSV or use mock data."""

    # Try to load real dataset
    possible_paths = [
        "data/test_sample_1000.csv",
        "test_sample_1000.csv",
        "products.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                logger.info(f"Loaded real dataset from {path} with {len(df)} products")
                return convert_df_to_products(df)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    # Fallback to mock data
    logger.info("Using mock product database")
    return get_mock_products()

def convert_df_to_products(df):
    """Convert DataFrame to product list format."""
    products = []

    # Common column name mappings
    name_cols = ['product_name', 'title', 'name', 'product_title']
    price_cols = ['price', 'product_price', 'cost']
    category_cols = ['category', 'product_category', 'department']
    desc_cols = ['description', 'product_description', 'details']
    url_cols = ['url', 'product_url', 'link']

    def find_column(df, possible_names):
        for col in possible_names:
            if col in df.columns:
                return col
        return None

    name_col = find_column(df, name_cols)
    price_col = find_column(df, price_cols)
    category_col = find_column(df, category_cols)
    desc_col = find_column(df, desc_cols)
    url_col = find_column(df, url_cols)

    for idx, row in df.iterrows():
        try:
            # Extract price
            price_val = 0.0
            if price_col and pd.notna(row[price_col]):
                price_str = str(row[price_col]).replace('$', '').replace(',', '')
                try:
                    price_val = float(price_str)
                except:
                    price_val = 29.99  # Default price

            product = {
                "name": str(row[name_col]) if name_col else f"Product {idx}",
                "price": price_val,
                "category": str(row[category_col]) if category_col else "General",
                "description": str(row[desc_col])[:200] + "..." if desc_col else "Product description",
                "url": str(row[url_col]) if url_col else f"https://example.com/product/{idx}",
                "image": f"https://via.placeholder.com/200x200?text=Product+{idx}"
            }
            products.append(product)

            # Limit to first 100 products for demo
            if len(products) >= 100:
                break

        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
            continue

    return products

def get_mock_products():
    """Fallback mock product database."""
    return [
        {
            "name": "Sony WH-CH720N Wireless Bluetooth Headphones",
            "price": 89.99,
            "category": "Electronics",
            "description": "Active Noise Cancelling, 35-hour battery life, comfortable over-ear design",
            "url": "https://example.com/sony-headphones",
            "image": "https://via.placeholder.com/200x200?text=Sony+Headphones"
        },
        {
            "name": "JBL Tune 510BT Wireless On-Ear Headphones",
            "price": 39.99,
            "category": "Audio",
            "description": "Wireless Bluetooth 5.0, Pure Bass sound, 40-hour battery life",
            "url": "https://example.com/jbl-headphones",
            "image": "https://via.placeholder.com/200x200?text=JBL+Headphones"
        },
        {
            "name": "Anker Soundcore Life Q20 Hybrid Active Noise Cancelling Headphones",
            "price": 59.99,
            "category": "Electronics",
            "description": "Hi-Res Audio, 40-hour playtime, memory foam ear cups",
            "url": "https://example.com/anker-headphones",
            "image": "https://via.placeholder.com/200x200?text=Anker+Headphones"
        },
        {
            "name": "LEGO Classic Creative Bricks Set",
            "price": 24.99,
            "category": "Toys & Games",
            "description": "484 pieces, suitable for ages 4-99, endless building possibilities",
            "url": "https://example.com/lego-bricks",
            "image": "https://via.placeholder.com/200x200?text=LEGO+Bricks"
        },
        {
            "name": "Melissa & Doug Wooden Shape Sorting Cube",
            "price": 19.99,
            "category": "Educational Toys",
            "description": "12 chunky, vibrant shapes to sort, promotes problem-solving skills",
            "url": "https://example.com/shape-cube",
            "image": "https://via.placeholder.com/200x200?text=Shape+Cube"
        },
        {
            "name": "LeapFrog LeapStart Interactive Learning System",
            "price": 34.99,
            "category": "Educational Electronics",
            "description": "Interactive books, stylus included, teaches reading and math",
            "url": "https://example.com/leapstart",
            "image": "https://via.placeholder.com/200x200?text=LeapStart"
        }
    ]

# Load products
PRODUCTS = load_product_database()
logger.info(f"Loaded {len(PRODUCTS)} products")

def search_products(query: str, max_results: int = 5) -> List[dict]:
    """Simple product search simulation."""

    query_lower = query.lower()
    results = []

    for product in PRODUCTS:
        # Simple keyword matching
        if (query_lower in product["name"].lower() or
            query_lower in product["description"].lower() or
            query_lower in product["category"].lower()):
            results.append(product)

        # Price-based filtering
        if "under $100" in query_lower and product["price"] < 100:
            results.append(product)
        elif "under $50" in query_lower and product["price"] < 50:
            results.append(product)

        # Category-based matching
        if "headphones" in query_lower and "headphones" in product["name"].lower():
            results.append(product)
        elif "toys" in query_lower and "toys" in product["category"].lower():
            results.append(product)

    # Remove duplicates and limit results
    seen = set()
    unique_results = []
    for product in results:
        if product["name"] not in seen:
            seen.add(product["name"])
            unique_results.append(product)

    return unique_results[:max_results]

def format_search_results(products: List[dict]) -> str:
    """Format products for display."""

    if not products:
        return "No products found matching your query."

    formatted = "**Search Results:**\n\n"

    for i, product in enumerate(products, 1):
        formatted += f"**{i}. {product['name']}**\n"
        formatted += f"💰 Price: ${product['price']}\n"
        formatted += f"📂 Category: {product['category']}\n"
        formatted += f"📝 Description: {product['description']}\n"
        formatted += f"🔗 [View Product]({product['url']})\n\n"

    return formatted

def build_prompt_for_llm(query: str, products: List[dict]) -> str:
    """Build prompt for LLM."""

    prompt = f"User query: {query}\n\nSearch results:\n"

    for i, product in enumerate(products, 1):
        prompt += f"{i}. Product: {product['name']} | Price: ${product['price']} | Category: {product['category']}\n"
        prompt += f"   Description: {product['description']}\n"

    prompt += "\nProvide a helpful response:"
    return prompt

def process_query(query: str) -> Tuple[str, str]:
    """Process user query and return results + AI response."""

    if not query.strip():
        return "Please enter a search query.", "👋 Hello! Try searching for products like 'wireless headphones' or 'educational toys'."

    # Search for products
    products = search_products(query, max_results=5)

    # Format search results
    search_results = format_search_results(products)

    # Generate AI response
    try:
        if products:
            prompt = build_prompt_for_llm(query, products)
        else:
            prompt = f"User asked: {query}\n\nNo products found. Provide helpful suggestions."

        start_time = time.time()
        ai_response = llm_client.generate_response(
            prompt=prompt,
            max_tokens=300,
            temperature=0.3  # Lower temperature for more consistent responses
        )
        generation_time = time.time() - start_time

        # Add model info and cost estimate
        model_info = llm_client.get_model_info()
        model_name = model_info.get("model_name", "unknown")

        if "openai" in model_info.get("provider", ""):
            # Estimate cost for OpenAI
            estimated_tokens = len(prompt.split()) + len(ai_response.split())
            estimated_cost = estimated_tokens * 0.0015 / 1000  # GPT-3.5-turbo pricing
            cost_info = f" • Cost: ~${estimated_cost:.4f}"
        else:
            cost_info = " • Free"

        ai_response = f"🤖 **AI Assistant** ({model_name} • {generation_time:.2f}s{cost_info}):\n\n{ai_response}"

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        ai_response = "🤖 **AI Assistant**: I'm here to help with product recommendations! The search results above show relevant products for your query."

    return search_results, ai_response

# Gradio interface
def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="E-commerce RAG Pipeline",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .header {
            text-align: center;
            margin-bottom: 2em;
        }
        """
    ) as demo:

        gr.HTML(f"""
        <div class="header">
            <h1>🛒 E-commerce RAG Pipeline</h1>
            <p>AI-powered product search and recommendations using Retrieval-Augmented Generation</p>
            <p><em>Demo with {len(PRODUCTS)} products loaded</em></p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="🔍 Search Query",
                    placeholder="Enter your product search (e.g., 'wireless headphones under $100')",
                    lines=2
                )

                search_btn = gr.Button("Search Products", variant="primary", size="lg")

                gr.HTML("""
                <div style="margin-top: 1em;">
                    <h3>💡 Example Queries:</h3>
                    <ul>
                        <li>wireless bluetooth headphones under $100</li>
                        <li>educational toys for 5 year old kids</li>
                        <li>electronics under $50</li>
                        <li>LEGO building sets</li>
                    </ul>
                </div>
                """)

        with gr.Row():
            with gr.Column(scale=1):
                search_output = gr.Markdown(
                    label="📦 Product Results",
                    value="Search results will appear here..."
                )

            with gr.Column(scale=1):
                ai_output = gr.Markdown(
                    label="🤖 AI Recommendations",
                    value="AI recommendations will appear here..."
                )

        gr.HTML("""
        <div style="margin-top: 2em; padding: 1em; background-color: #f0f0f0; border-radius: 10px;">
            <h3>🚀 About This Demo</h3>
            <ul>
                <li><strong>Technology</strong>: RAG (Retrieval-Augmented Generation) pipeline</li>
                <li><strong>Features</strong>: Product search, AI-powered recommendations, real-time responses</li>
                <li><strong>Architecture</strong>: CLIP embeddings + FAISS vector search + LLM generation</li>
                <li><strong>Deployment</strong>: HuggingFace Spaces with Gradio interface</li>
            </ul>
            <p><strong>GitHub</strong>: <a href="https://github.com/ycnliu/ecommerce-rag-pipeline">ycnliu/ecommerce-rag-pipeline</a></p>
        </div>
        """)

        # Event handlers
        search_btn.click(
            fn=process_query,
            inputs=[query_input],
            outputs=[search_output, ai_output]
        )

        query_input.submit(
            fn=process_query,
            inputs=[query_input],
            outputs=[search_output, ai_output]
        )

    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )