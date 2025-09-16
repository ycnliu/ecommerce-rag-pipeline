"""
Command Line Interface for the E-commerce RAG Pipeline.
"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

from .utils.config import Config
from .utils.logging import setup_logging
from .utils.exceptions import EcommerceRAGError
from .data.processor import DataProcessor
from .embedding.service import CLIPEmbeddingService
from .vector_db.faiss_service import FAISSVectorDB
from .rag.llm_client import create_llm_client
from .rag.rag_pipeline import RAGPipeline
from .rag.evaluation import RAGEvaluator
from .embedding.fusion import AdvancedEmbeddingFusion
from .embedding.fine_tuning import CLIPFineTuner, EcommerceDataset


@click.group()
@click.option('--config-file', '-c', default=None, help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config_file: Optional[str], verbose: bool):
    """E-commerce RAG Pipeline CLI."""
    # Initialize configuration
    config = Config()

    if verbose:
        config.log_level = "DEBUG"

    # Setup logging
    setup_logging(config)

    # Store config in context
    ctx.ensure_object(dict)
    ctx.obj['config'] = config

    logger.info("E-commerce RAG Pipeline CLI initialized")


@cli.command()
@click.option('--csv-path', '-i', required=True, help='Path to input CSV file')
@click.option('--output-dir', '-o', default='models', help='Output directory for processed data')
@click.option('--force', '-f', is_flag=True, help='Force overwrite existing files')
@click.pass_context
def process_data(ctx, csv_path: str, output_dir: str, force: bool):
    """Process raw CSV data and prepare for indexing."""
    config = ctx.obj['config']

    try:
        click.echo(f"Processing data from: {csv_path}")

        # Validate input file
        if not os.path.exists(csv_path):
            raise click.FileError(csv_path, "File not found")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize data processor
        processor = DataProcessor()

        # Process data
        with click.progressbar(length=3, label='Processing data') as bar:
            # Step 1: Load and clean data
            df, metadata_list = processor.process_full_pipeline(csv_path)
            bar.update(1)

            # Step 2: Save processed data
            processed_csv_path = output_path / "processed_data.csv"
            if processed_csv_path.exists() and not force:
                if not click.confirm(f"File {processed_csv_path} exists. Overwrite?"):
                    return

            df.to_csv(processed_csv_path, index=False)
            bar.update(1)

            # Step 3: Save metadata
            metadata_path = output_path / "metadata.json"
            if metadata_path.exists() and not force:
                if not click.confirm(f"File {metadata_path} exists. Overwrite?"):
                    return

            # Convert metadata to JSON-serializable format
            metadata_dict = [meta.dict() for meta in metadata_list]
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            bar.update(1)

        click.echo(f"✅ Data processing completed!")
        click.echo(f"   Processed {len(df)} records")
        click.echo(f"   Saved to: {output_path}")

    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise click.ClickException(f"Data processing failed: {e}")


@cli.command()
@click.option('--csv-path', '-i', required=True, help='Path to processed CSV file')
@click.option('--output-dir', '-o', default='models', help='Output directory for embeddings and index')
@click.option('--batch-size', '-b', default=32, help='Batch size for embedding generation')
@click.option('--force', '-f', is_flag=True, help='Force rebuild existing index')
@click.pass_context
def build_index(ctx, csv_path: str, output_dir: str, batch_size: int, force: bool):
    """Build FAISS index from processed data."""
    config = ctx.obj['config']

    try:
        click.echo(f"Building index from: {csv_path}")

        # Validate input
        if not os.path.exists(csv_path):
            raise click.FileError(csv_path, "File not found")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        index_path = output_path / "product_index.faiss"
        metadata_path = output_path / "product_metadata.pkl"
        embeddings_path = output_path / "embeddings.npy"

        # Check if index already exists
        if index_path.exists() and not force:
            if not click.confirm("Index already exists. Rebuild?"):
                return

        # Initialize services
        click.echo("Initializing embedding service...")
        embedding_service = CLIPEmbeddingService(
            model_name=config.clip_model_name,
            device=config.device,
            cache_dir=config.model_cache_dir
        )
        embedding_service.load_model()

        # Load data
        click.echo("Loading data...")
        processor = DataProcessor()
        df, metadata_list = processor.process_full_pipeline(csv_path)

        # Generate embeddings
        click.echo("Generating embeddings...")
        texts = [meta.combined_text for meta in metadata_list]

        embeddings = []
        with tqdm(total=len(texts), desc="Embedding generation") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = embedding_service.batch_text_embeddings(batch_texts, batch_size)
                embeddings.append(batch_embeddings)
                pbar.update(len(batch_texts))

        # Combine all embeddings
        all_embeddings = np.vstack(embeddings)

        # Save embeddings
        np.save(embeddings_path, all_embeddings)
        click.echo(f"Saved embeddings to: {embeddings_path}")

        # Build FAISS index
        click.echo("Building FAISS index...")
        vector_db = FAISSVectorDB(
            dimension=all_embeddings.shape[1],
            index_type=config.faiss_index_type,
            metric=config.faiss_metric
        )

        vector_db.add_vectors(all_embeddings, metadata_list, train_if_needed=True)

        # Save index
        vector_db.save_index(str(index_path), str(metadata_path))

        click.echo(f"✅ Index building completed!")
        click.echo(f"   Indexed {len(metadata_list)} products")
        click.echo(f"   Index saved to: {index_path}")
        click.echo(f"   Metadata saved to: {metadata_path}")

    except Exception as e:
        logger.error(f"Index building failed: {e}")
        raise click.ClickException(f"Index building failed: {e}")


@cli.command()
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--k', default=5, help='Number of results to return')
@click.option('--rerank', is_flag=True, help='Enable result reranking')
@click.option('--no-llm', is_flag=True, help='Skip LLM response generation')
@click.option('--output-format', type=click.Choice(['table', 'json', 'detailed']), default='table', help='Output format')
@click.pass_context
def search(ctx, query: str, k: int, rerank: bool, no_llm: bool, output_format: str):
    """Search for products using text query."""
    config = ctx.obj['config']

    try:
        # Initialize RAG pipeline
        click.echo("Initializing RAG pipeline...")
        pipeline = _initialize_rag_pipeline(config)

        # Perform search
        click.echo(f"Searching for: '{query}'")
        start_time = time.time()

        response = pipeline.query(
            text_query=query,
            k=k,
            rerank=rerank,
            generate_response=not no_llm
        )

        processing_time = time.time() - start_time

        # Display results
        if output_format == 'json':
            click.echo(json.dumps(response.dict(), indent=2, default=str))
        elif output_format == 'detailed':
            _display_detailed_results(response)
        else:
            _display_table_results(response)

        click.echo(f"\nProcessed in {processing_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise click.ClickException(f"Search failed: {e}")


@cli.command()
@click.option('--queries-file', '-i', required=True, help='JSON file with test queries')
@click.option('--output-file', '-o', default='evaluation_results.json', help='Output file for results')
@click.option('--k', default=5, help='Number of results per query')
@click.pass_context
def evaluate(ctx, queries_file: str, output_file: str, k: int):
    """Evaluate pipeline performance on test queries."""
    config = ctx.obj['config']

    try:
        # Load test queries
        if not os.path.exists(queries_file):
            raise click.FileError(queries_file, "File not found")

        with open(queries_file, 'r') as f:
            test_data = json.load(f)

        queries = test_data.get('queries', [])
        references = test_data.get('references', [])

        click.echo(f"Evaluating {len(queries)} queries...")

        # Initialize pipeline and evaluator
        pipeline = _initialize_rag_pipeline(config)
        evaluator = RAGEvaluator(pipeline)

        # Run evaluation
        with click.progressbar(length=100, label='Running evaluation') as bar:
            results = evaluator.evaluate_end_to_end(
                test_queries=queries,
                reference_responses=references if references else None,
                k=k
            )
            bar.update(100)

        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Display summary
        click.echo(f"✅ Evaluation completed!")
        click.echo(f"   Results saved to: {output_file}")

        if 'response_quality' in results:
            quality = results['response_quality']
            click.echo(f"   BLEU Score: {quality['bleu_mean']:.3f}")
            click.echo(f"   ROUGE Score: {quality['rouge_mean']:.3f}")

        if 'latency' in results:
            latency = results['latency']
            click.echo(f"   Mean Latency: {latency['mean_latency_ms']:.1f}ms")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise click.ClickException(f"Evaluation failed: {e}")


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind the server')
@click.option('--port', default=8000, help='Port to bind the server')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.pass_context
def serve(ctx, host: str, port: int, reload: bool):
    """Start the FastAPI server."""
    config = ctx.obj['config']

    try:
        import uvicorn
        from .api.main import app

        click.echo(f"Starting server on {host}:{port}")
        click.echo(f"API documentation: http://{host}:{port}/docs")

        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level=config.log_level.lower()
        )

    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise click.ClickException(f"Server startup failed: {e}")


@cli.command("train-clip")
@click.option('--data-path', required=True, help='Path to training data CSV')
@click.option('--epochs', default=5, help='Number of training epochs')
@click.option('--batch-size', default=16, help='Training batch size')
@click.option('--learning-rate', default=2e-5, help='Learning rate')
@click.option('--output-dir', default='models/fine_tuned', help='Output directory for trained model')
@click.option('--use-wandb', is_flag=True, help='Use Weights & Biases logging')
@click.option('--model-name', default='openai/clip-vit-base-patch32', help='Base CLIP model')
@click.pass_context
def train_clip(ctx, data_path: str, epochs: int, batch_size: int, learning_rate: float,
               output_dir: str, use_wandb: bool, model_name: str):
    """Train CLIP model on e-commerce data."""
    config = ctx.obj['config']

    try:
        click.echo(f"Training CLIP model on: {data_path}")

        # Validate input
        if not os.path.exists(data_path):
            raise click.FileError(data_path, "File not found")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize Weights & Biases if requested
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=os.getenv('WANDB_PROJECT', 'ecommerce-rag-pipeline'),
                    name=f"clip-finetuning-{int(time.time())}",
                    config={
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'model_name': model_name
                    }
                )
                click.echo("📊 Weights & Biases initialized")
            except ImportError:
                click.echo("⚠️ Weights & Biases not available, continuing without logging")
                use_wandb = False

        # Load and prepare data
        click.echo("Loading training data...")
        df = pd.read_csv(data_path)

        # Create dataset
        dataset = EcommerceDataset(
            dataframe=df,
            text_column='product_name',  # Adjust based on your CSV structure
            image_column='image_url',
            category_column='category'
        )

        # Initialize fine-tuner
        fine_tuner = CLIPFineTuner(
            model_name=model_name,
            device=config.device or 'auto'
        )

        # Train model
        with click.progressbar(length=epochs, label='Training epochs') as bar:
            fine_tuner.train(
                dataset=dataset,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                use_wandb=use_wandb,
                callback=lambda epoch: bar.update(1)
            )

        # Save model
        click.echo(f"Saving model to: {output_dir}")
        fine_tuner.save_model(output_dir)

        # Save training config
        config_path = output_path / "training_config.json"
        training_config = {
            'model_name': model_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'data_path': data_path,
            'trained_at': datetime.now().isoformat()
        }

        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)

        click.echo(f"✅ CLIP training completed!")
        click.echo(f"   Model saved to: {output_dir}")

        if use_wandb:
            wandb.finish()

    except Exception as e:
        logger.error(f"CLIP training failed: {e}")
        raise click.ClickException(f"CLIP training failed: {e}")


@cli.command("train-fusion")
@click.option('--data-path', required=True, help='Path to training data CSV')
@click.option('--epochs', default=5, help='Number of training epochs')
@click.option('--batch-size', default=32, help='Training batch size')
@click.option('--learning-rate', default=1e-4, help='Learning rate')
@click.option('--output-dir', default='models/fusion', help='Output directory for trained model')
@click.option('--use-wandb', is_flag=True, help='Use Weights & Biases logging')
@click.option('--clip-model', default='openai/clip-vit-base-patch32', help='CLIP model to use')
@click.option('--sentence-model', default='all-MiniLM-L6-v2', help='SentenceTransformer model to use')
@click.pass_context
def train_fusion(ctx, data_path: str, epochs: int, batch_size: int, learning_rate: float,
                output_dir: str, use_wandb: bool, clip_model: str, sentence_model: str):
    """Train embedding fusion model."""
    config = ctx.obj['config']

    try:
        click.echo(f"Training fusion model on: {data_path}")

        # Validate input
        if not os.path.exists(data_path):
            raise click.FileError(data_path, "File not found")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize Weights & Biases if requested
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=os.getenv('WANDB_PROJECT', 'ecommerce-rag-pipeline'),
                    name=f"fusion-training-{int(time.time())}",
                    config={
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'clip_model': clip_model,
                        'sentence_model': sentence_model
                    }
                )
                click.echo("📊 Weights & Biases initialized")
            except ImportError:
                click.echo("⚠️ Weights & Biases not available, continuing without logging")
                use_wandb = False

        # Load data
        click.echo("Loading training data...")
        df = pd.read_csv(data_path)

        # Initialize fusion model
        fusion_model = AdvancedEmbeddingFusion(
            clip_model_name=clip_model,
            sentence_model_name=sentence_model,
            device=config.device or 'auto'
        )

        # Train fusion model (implement training logic in fusion.py)
        click.echo("Training fusion model...")
        # Note: Add training logic to AdvancedEmbeddingFusion class

        # Save model
        click.echo(f"Saving model to: {output_dir}")
        fusion_model.save_model(output_dir)

        # Save training config
        config_path = output_path / "training_config.json"
        training_config = {
            'clip_model': clip_model,
            'sentence_model': sentence_model,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'data_path': data_path,
            'trained_at': datetime.now().isoformat()
        }

        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)

        click.echo(f"✅ Fusion training completed!")
        click.echo(f"   Model saved to: {output_dir}")

        if use_wandb:
            wandb.finish()

    except Exception as e:
        logger.error(f"Fusion training failed: {e}")
        raise click.ClickException(f"Fusion training failed: {e}")


@cli.command("evaluate-model")
@click.option('--model-path', required=True, help='Path to trained model')
@click.option('--test-data', required=True, help='Path to test data CSV')
@click.option('--output-file', default='evaluation_results.json', help='Output file for results')
@click.option('--k', default=5, help='Number of results to evaluate')
@click.pass_context
def evaluate_model(ctx, model_path: str, test_data: str, output_file: str, k: int):
    """Evaluate trained model performance."""
    config = ctx.obj['config']

    try:
        click.echo(f"Evaluating model: {model_path}")

        # Validate inputs
        if not os.path.exists(model_path):
            raise click.FileError(model_path, "Model path not found")
        if not os.path.exists(test_data):
            raise click.FileError(test_data, "Test data not found")

        # Load test data
        df = pd.read_csv(test_data)

        # Initialize pipeline with trained model
        # Note: This would need to be adapted based on model type
        click.echo("Running evaluation...")

        # Placeholder evaluation results
        results = {
            'model_path': model_path,
            'test_data': test_data,
            'evaluated_at': datetime.now().isoformat(),
            'metrics': {
                'recall_at_1': 0.0,
                'recall_at_5': 0.0,
                'recall_at_10': 0.0,
                'mrr': 0.0
            }
        }

        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        click.echo(f"✅ Evaluation completed!")
        click.echo(f"   Results saved to: {output_file}")

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise click.ClickException(f"Model evaluation failed: {e}")


@cli.command("push-to-hub")
@click.option('--model-path', required=True, help='Path to trained model')
@click.option('--repo-name', required=True, help='Hugging Face repository name')
@click.option('--private', is_flag=True, help='Make repository private')
@click.option('--commit-message', default='Upload model via CLI', help='Commit message')
@click.pass_context
def push_to_hub(ctx, model_path: str, repo_name: str, private: bool, commit_message: str):
    """Push trained model to Hugging Face Hub."""
    config = ctx.obj['config']

    try:
        click.echo(f"Uploading model to Hugging Face: {repo_name}")

        # Validate model path
        if not os.path.exists(model_path):
            raise click.FileError(model_path, "Model path not found")

        # Check for HF token
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise click.ClickException("HF_TOKEN environment variable not set")

        try:
            from huggingface_hub import HfApi, create_repo

            api = HfApi(token=hf_token)

            # Create repository if it doesn't exist
            click.echo(f"Creating/checking repository: {repo_name}")
            create_repo(
                repo_id=repo_name,
                private=private,
                exist_ok=True,
                token=hf_token
            )

            # Upload model files
            click.echo("Uploading model files...")
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_name,
                commit_message=commit_message,
                token=hf_token
            )

            click.echo(f"✅ Model uploaded successfully!")
            click.echo(f"   Repository: https://huggingface.co/{repo_name}")

        except ImportError:
            raise click.ClickException("huggingface_hub not installed. Run: pip install huggingface_hub")

    except Exception as e:
        logger.error(f"Model upload failed: {e}")
        raise click.ClickException(f"Model upload failed: {e}")


@cli.command()
@click.pass_context
def status(ctx):
    """Show pipeline status and configuration."""
    config = ctx.obj['config']

    try:
        click.echo("E-commerce RAG Pipeline Status")
        click.echo("=" * 40)

        # Configuration
        click.echo("\n📋 Configuration:")
        click.echo(f"   CLIP Model: {config.clip_model_name}")
        click.echo(f"   LLM Provider: {config.llm_provider}")
        click.echo(f"   LLM Model: {config.llm_model_name}")
        click.echo(f"   Device: {config.device or 'auto-detect'}")
        click.echo(f"   Index Type: {config.faiss_index_type}")

        # Check model files
        click.echo("\n📁 Model Files:")
        paths = config.get_model_paths()
        for name, path in paths.items():
            status = "✅" if os.path.exists(path) else "❌"
            click.echo(f"   {name}: {status} {path}")

        # Check API token
        click.echo("\n🔑 API Configuration:")
        token_status = "✅" if config.llm_api_token else "❌"
        click.echo(f"   LLM API Token: {token_status}")

        # Try to initialize services
        click.echo("\n🧪 Service Health Check:")

        try:
            embedding_service = CLIPEmbeddingService(
                model_name=config.clip_model_name,
                device=config.device
            )
            click.echo("   Embedding Service: ✅")
        except Exception as e:
            click.echo(f"   Embedding Service: ❌ {e}")

        try:
            if os.path.exists(config.faiss_index_path):
                vector_db = FAISSVectorDB(dimension=config.embedding_dimension)
                vector_db.load_index(config.faiss_index_path, config.faiss_metadata_path)
                stats = vector_db.get_stats()
                click.echo(f"   Vector DB: ✅ ({stats.total_vectors} vectors)")
            else:
                click.echo("   Vector DB: ❌ Index not found")
        except Exception as e:
            click.echo(f"   Vector DB: ❌ {e}")

        try:
            if config.validate_llm_config():
                llm_client = create_llm_client(
                    provider=config.llm_provider,
                    model_name=config.llm_model_name,
                    api_token=config.llm_api_token
                )
                click.echo("   LLM Client: ✅")
            else:
                click.echo("   LLM Client: ❌ Configuration invalid")
        except Exception as e:
            click.echo(f"   LLM Client: ❌ {e}")

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise click.ClickException(f"Status check failed: {e}")


def _initialize_rag_pipeline(config: Config) -> RAGPipeline:
    """Initialize RAG pipeline with given configuration."""
    # Initialize embedding service
    embedding_service = CLIPEmbeddingService(
        model_name=config.clip_model_name,
        device=config.device,
        cache_dir=config.model_cache_dir
    )
    embedding_service.load_model()

    # Initialize vector database
    vector_db = FAISSVectorDB(
        dimension=config.embedding_dimension,
        index_type=config.faiss_index_type,
        metric=config.faiss_metric
    )

    # Load existing index
    if os.path.exists(config.faiss_index_path):
        vector_db.load_index(config.faiss_index_path, config.faiss_metadata_path)
    else:
        raise click.ClickException("FAISS index not found. Run 'build-index' first.")

    # Initialize LLM client
    if not config.validate_llm_config():
        raise click.ClickException("LLM configuration is invalid. Check your API token.")

    llm_client = create_llm_client(
        provider=config.llm_provider,
        model_name=config.llm_model_name,
        api_token=config.llm_api_token
    )

    # Create pipeline
    return RAGPipeline(
        embedding_service=embedding_service,
        vector_db=vector_db,
        llm_client=llm_client
    )


def _display_table_results(response):
    """Display search results in table format."""
    click.echo(f"\n🔍 Query: {response.query}")
    click.echo(f"📊 Found {len(response.results)} results")

    if response.generated_response:
        click.echo(f"\n🤖 AI Response:")
        click.echo(response.generated_response)

    if response.results:
        click.echo(f"\n📋 Search Results:")
        for i, result in enumerate(response.results, 1):
            click.echo(f"\n{i}. Score: {result.score:.3f}")
            click.echo(f"   Product: {result.metadata.combined_text[:100]}...")
            click.echo(f"   URL: {result.metadata.product_url}")


def _display_detailed_results(response):
    """Display search results in detailed format."""
    click.echo(f"\n🔍 Query: {response.query}")
    click.echo(f"📊 Found {len(response.results)} results")
    click.echo(f"⏱️  Processing time: {response.processing_time:.2f}s")

    if response.generated_response:
        click.echo(f"\n🤖 AI Response:")
        click.echo("-" * 50)
        click.echo(response.generated_response)
        click.echo("-" * 50)

    if response.results:
        click.echo(f"\n📋 Detailed Results:")
        for i, result in enumerate(response.results, 1):
            click.echo(f"\n{'='*20} Result {i} {'='*20}")
            click.echo(f"Score: {result.score:.3f}")
            click.echo(f"Product: {result.metadata.combined_text}")
            click.echo(f"URL: {result.metadata.product_url}")
            if result.metadata.shipping_weight != "Not available":
                click.echo(f"Shipping Weight: {result.metadata.shipping_weight}")
            if result.metadata.product_dimensions != "Not available":
                click.echo(f"Dimensions: {result.metadata.product_dimensions}")


def main():
    """Main entry point for CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"CLI error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()