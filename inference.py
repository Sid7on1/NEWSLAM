"""
SLAM3 Inference Script for Cloud Deployment
Supports: REST API, CLI, and Batch Processing
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from slam3_pytorch import SLAM3Model, SLAM3Config
from google.cloud import storage
import argparse
import json
import tempfile
from flask import Flask, request, jsonify
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════

class SLAM3Inference:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        gcs_bucket: Optional[str] = None,
        gcs_checkpoint: Optional[str] = "checkpoints/best.pt",
        device: Optional[str] = None
    ):
        """
        Initialize SLAM3 for inference

        Args:
            checkpoint_path: Local path to checkpoint
            gcs_bucket: GCS bucket name (if loading from cloud)
            gcs_checkpoint: Path to checkpoint in GCS bucket
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize model
        logger.info("Initializing model...")
        self.config = SLAM3Config()
        self.model = SLAM3Model(self.config)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load checkpoint
        if checkpoint_path:
            self._load_checkpoint_local(checkpoint_path)
        elif gcs_bucket:
            self._load_checkpoint_gcs(gcs_bucket, gcs_checkpoint)
        else:
            logger.warning("No checkpoint provided, using randomly initialized model")

        logger.info("✅ Model ready for inference")

    def _load_checkpoint_local(self, path: str):
        """Load checkpoint from local file"""
        logger.info(f"Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")

    def _load_checkpoint_gcs(self, bucket_name: str, checkpoint_path: str):
        """Load checkpoint from GCS"""
        logger.info(f"Loading checkpoint from GCS: gs://{bucket_name}/{checkpoint_path}")

        # Download checkpoint
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(checkpoint_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            blob.download_to_filename(tmp.name)
            checkpoint = torch.load(tmp.name, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text from prompt

        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated texts
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        results = []
        for _ in range(num_return_sequences):
            current_ids = input_ids.clone()

            for _ in range(max_length):
                # Truncate if too long
                if current_ids.shape[1] > self.config.SEQ_LEN:
                    current_ids = current_ids[:, -self.config.SEQ_LEN:]

                # Get logits
                logits = self.model(current_ids, training=False)
                logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                # Append
                current_ids = torch.cat([current_ids, next_token], dim=1)

                # Stop on EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            # Decode
            generated_text = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
            results.append(generated_text)

        return results

    def get_perplexity(self, text: str) -> float:
        """Calculate perplexity of text"""
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)

        if input_ids.shape[1] <= 1:
            return float('inf')

        # Split into input and target
        tokens = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        # Truncate if needed
        if tokens.shape[1] > self.config.SEQ_LEN:
            tokens = tokens[:, :self.config.SEQ_LEN]
            targets = targets[:, :self.config.SEQ_LEN]

        # Get logits
        logits = self.model(tokens, training=False)

        # Calculate loss
        logits_flat = logits.reshape(-1, self.config.VOCAB_SIZE)
        targets_flat = targets.reshape(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)

        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()

        return perplexity

# ═══════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def cli_interface():
    """Command-line interface for inference"""
    parser = argparse.ArgumentParser(description='SLAM3 Inference')

    # Model loading
    parser.add_argument('--checkpoint', type=str, help='Local checkpoint path')
    parser.add_argument('--gcs-bucket', type=str, help='GCS bucket name')
    parser.add_argument('--gcs-checkpoint', type=str, default='checkpoints/best.pt',
                       help='Checkpoint path in GCS bucket')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use')

    # Generation parameters
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    parser.add_argument('--max-length', type=int, default=100, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.95, help='Nucleus sampling')
    parser.add_argument('--num-sequences', type=int, default=1, help='Number of sequences')

    # Output
    parser.add_argument('--output', type=str, help='Output file (JSON)')

    args = parser.parse_args()

    # Initialize model
    model = SLAM3Inference(
        checkpoint_path=args.checkpoint,
        gcs_bucket=args.gcs_bucket,
        gcs_checkpoint=args.gcs_checkpoint,
        device=args.device
    )

    # Generate
    logger.info(f"Generating with prompt: {args.prompt}")
    results = model.generate(
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_sequences
    )

    # Output
    output_data = {
        'prompt': args.prompt,
        'generated': results,
        'parameters': {
            'max_length': args.max_length,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
        }
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"✅ Saved output to: {args.output}")
    else:
        print("\n" + "="*80)
        print("GENERATED TEXT:")
        print("="*80)
        for i, text in enumerate(results, 1):
            print(f"\n[{i}] {text}")
        print("\n" + "="*80)

# ═══════════════════════════════════════════════════════════════════════════
# REST API INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
inference_model = None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': inference_model is not None})

@app.route('/generate', methods=['POST'])
def generate_api():
    """
    Generate text endpoint

    Request body:
    {
        "prompt": "Once upon a time",
        "max_length": 100,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
        "num_sequences": 1
    }
    """
    if inference_model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'Missing prompt'}), 400

        # Generate
        results = inference_model.generate(
            prompt=prompt,
            max_length=data.get('max_length', 100),
            temperature=data.get('temperature', 0.8),
            top_k=data.get('top_k', 50),
            top_p=data.get('top_p', 0.95),
            num_return_sequences=data.get('num_sequences', 1)
        )

        return jsonify({
            'prompt': prompt,
            'generated': results,
            'success': True
        })

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/perplexity', methods=['POST'])
def perplexity_api():
    """
    Calculate perplexity endpoint

    Request body:
    {
        "text": "Some text to evaluate"
    }
    """
    if inference_model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        text = data.get('text')

        if not text:
            return jsonify({'error': 'Missing text'}), 400

        perplexity = inference_model.get_perplexity(text)

        return jsonify({
            'text': text,
            'perplexity': perplexity,
            'success': True
        })

    except Exception as e:
        logger.error(f"Error calculating perplexity: {e}")
        return jsonify({'error': str(e)}), 500

def run_api_server(
    host: str = '0.0.0.0',
    port: int = 8080,
    checkpoint_path: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    gcs_checkpoint: str = 'checkpoints/best.pt'
):
    """Run REST API server"""
    global inference_model

    logger.info("Initializing model for API server...")
    inference_model = SLAM3Inference(
        checkpoint_path=checkpoint_path,
        gcs_bucket=gcs_bucket,
        gcs_checkpoint=gcs_checkpoint
    )

    logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port)

# ═══════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def batch_generate(
    input_file: str,
    output_file: str,
    checkpoint_path: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    gcs_checkpoint: str = 'checkpoints/best.pt',
    **generation_kwargs
):
    """
    Process batch of prompts from file

    Args:
        input_file: JSON file with prompts (list of strings or list of dicts)
        output_file: Output JSON file
        checkpoint_path: Local checkpoint path
        gcs_bucket: GCS bucket name
        gcs_checkpoint: Checkpoint path in bucket
        **generation_kwargs: Additional generation parameters
    """
    # Load model
    model = SLAM3Inference(
        checkpoint_path=checkpoint_path,
        gcs_bucket=gcs_bucket,
        gcs_checkpoint=gcs_checkpoint
    )

    # Load prompts
    logger.info(f"Loading prompts from: {input_file}")
    with open(input_file, 'r') as f:
        prompts_data = json.load(f)

    # Extract prompts
    if isinstance(prompts_data, list):
        if isinstance(prompts_data[0], str):
            prompts = prompts_data
        elif isinstance(prompts_data[0], dict):
            prompts = [p['prompt'] for p in prompts_data]
    else:
        raise ValueError("Input file must contain a list of strings or dicts")

    logger.info(f"Processing {len(prompts)} prompts...")

    # Generate
    results = []
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"Processing {i}/{len(prompts)}: {prompt[:50]}...")

        generated = model.generate(prompt=prompt, **generation_kwargs)

        results.append({
            'prompt': prompt,
            'generated': generated,
            'index': i - 1
        })

    # Save results
    logger.info(f"Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("✅ Batch processing complete!")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'api':
        # Run API server
        parser = argparse.ArgumentParser(description='SLAM3 API Server')
        parser.add_argument('mode', choices=['api'])
        parser.add_argument('--checkpoint', type=str, help='Local checkpoint path')
        parser.add_argument('--gcs-bucket', type=str, help='GCS bucket name')
        parser.add_argument('--gcs-checkpoint', type=str, default='checkpoints/best.pt')
        parser.add_argument('--host', type=str, default='0.0.0.0')
        parser.add_argument('--port', type=int, default=8080)

        args = parser.parse_args()
        run_api_server(
            host=args.host,
            port=args.port,
            checkpoint_path=args.checkpoint,
            gcs_bucket=args.gcs_bucket,
            gcs_checkpoint=args.gcs_checkpoint
        )

    elif len(sys.argv) > 1 and sys.argv[1] == 'batch':
        # Batch processing
        parser = argparse.ArgumentParser(description='SLAM3 Batch Processing')
        parser.add_argument('mode', choices=['batch'])
        parser.add_argument('--input', type=str, required=True, help='Input JSON file')
        parser.add_argument('--output', type=str, required=True, help='Output JSON file')
        parser.add_argument('--checkpoint', type=str, help='Local checkpoint path')
        parser.add_argument('--gcs-bucket', type=str, help='GCS bucket name')
        parser.add_argument('--gcs-checkpoint', type=str, default='checkpoints/best.pt')
        parser.add_argument('--max-length', type=int, default=100)
        parser.add_argument('--temperature', type=float, default=0.8)

        args = parser.parse_args()
        batch_generate(
            input_file=args.input,
            output_file=args.output,
            checkpoint_path=args.checkpoint,
            gcs_bucket=args.gcs_bucket,
            gcs_checkpoint=args.gcs_checkpoint,
            max_length=args.max_length,
            temperature=args.temperature
        )

    else:
        # CLI interface
        cli_interface()
