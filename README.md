# 🚀 SLAM3: Multi-Level Mixture-of-Experts Language Model

A sophisticated language model architecture featuring:
- **3-Level Processing**: Memory Creation → Iterative Refinement → Final Synthesis
- **Mixture of Experts (MoE)**: Sparse expert routing for efficiency
- **Multi-Rotation Processing**: Multiple perspectives per token (4 rotations × 4 specialized blocks)
- **Specialized Blocks**: Structure, Patterns, Transitions, and Details
- **KV Caching**: Efficient attention with cached keys and values

## 📊 Model Architecture

```
Level 1: Memory Creation (Compute & Cache K/V)
    ↓
Level 2: Iterative Multi-Perspective Reasoning
    • EF Cycle 1: 4 rotations × 4 blocks = 16 views
    • EF Cycle 2: Another 16 views with updated embeddings
    • (Uses cached K/V from Level 1)
    ↓
Level 3: Final Synthesis
    ↓
Output Projection
```

### Key Features
- **RoPE Positional Encoding**: Rotary positional embeddings
- **Load Balancing**: Ensures uniform expert usage
- **Specialization Loss**: Encourages blocks to focus on different frequency patterns
- **Early Stopping**: Adaptive cycle termination during inference

## 🛠️ Installation

### Local Setup
```bash
git clone https://github.com/yourusername/slam3-training.git
cd slam3-training

# Install dependencies
pip install torch torchvision torchaudio transformers datasets
```

### Google Colab Setup
1. Upload `train_colab.ipynb` to Google Colab
2. Follow the notebook instructions
3. All dependencies are installed automatically

## 📚 Training on Google Colab

### Prerequisites
1. **Google Cloud Account** - For checkpoint storage
2. **GitHub Account** - For version control
3. **GPU/TPU Runtime** - Change runtime type in Colab

### Quick Start

1. **Open the Notebook**
   ```
   Upload train_colab.ipynb to Google Colab
   ```

2. **Run All Cells**
   The notebook will guide you through:
   - Authentication (GCP, GitHub, WandB)
   - GCS bucket creation/setup
   - GitHub repository creation/setup
   - Dataset download (TinyStories)
   - Model training with auto-save

3. **Training Features**
   - ✅ **Auto-save**: Checkpoints saved to GCS every 500 steps
   - ✅ **Auto-resume**: Automatically resumes from last checkpoint
   - ✅ **GitHub sync**: Code and logs pushed automatically
   - ✅ **Graceful shutdown**: Press Ctrl+C to save and stop
   - ✅ **Disconnect protection**: All progress saved to cloud

### Configuration

Model configuration in `slam3_pytorch.py`:

```python
class SLAM3Config:
    EMBED_DIM = 128          # Embedding dimension
    SEQ_LEN = 128            # Sequence length
    NUM_HEADS = 4            # Attention heads
    NUM_EXPERTS = 4          # Number of experts
    FF_DIM = 256             # Feed-forward dimension
    TOP_K_EXPERTS = 2        # Active experts per token
    NUM_ROTATIONS = 4        # Rotation count
    MAX_EF_CYCLES = 2        # EF cycles (reduced for speed)

    LEARNING_RATE = 3e-4
    BATCH_SIZE = 16
    DROPOUT = 0.1
    GRAD_CLIP = 1.0
    WARMUP_STEPS = 500
    MAX_STEPS = 50000
```

## 💾 Checkpoint Management

### GCS Structure
```
gs://your-bucket-name/
├── checkpoints/
│   ├── checkpoint_step_500.pt
│   ├── checkpoint_step_1000.pt
│   ├── latest.pt          # Latest checkpoint
│   └── best.pt            # Best validation loss
```

### Manual Checkpoint Operations

**Download checkpoint:**
```bash
gsutil cp gs://your-bucket/checkpoints/best.pt ./
```

**Upload checkpoint:**
```bash
gsutil cp checkpoint.pt gs://your-bucket/checkpoints/
```

**List checkpoints:**
```bash
gsutil ls gs://your-bucket/checkpoints/
```

## 🔮 Inference

### Command-Line Interface

**Basic generation:**
```bash
python inference.py \
  --checkpoint checkpoint.pt \
  --prompt "Once upon a time" \
  --max-length 100 \
  --temperature 0.8
```

**From GCS:**
```bash
python inference.py \
  --gcs-bucket your-bucket-name \
  --gcs-checkpoint checkpoints/best.pt \
  --prompt "The little girl" \
  --max-length 150
```

**Multiple sequences:**
```bash
python inference.py \
  --checkpoint checkpoint.pt \
  --prompt "In a magical forest" \
  --num-sequences 3 \
  --temperature 0.9
```

### REST API Server

**Start server:**
```bash
python inference.py api \
  --gcs-bucket your-bucket-name \
  --gcs-checkpoint checkpoints/best.pt \
  --host 0.0.0.0 \
  --port 8080
```

**API Endpoints:**

1. **Health Check**
   ```bash
   curl http://localhost:8080/health
   ```

2. **Generate Text**
   ```bash
   curl -X POST http://localhost:8080/generate \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Once upon a time",
       "max_length": 100,
       "temperature": 0.8,
       "top_k": 50,
       "top_p": 0.95
     }'
   ```

3. **Calculate Perplexity**
   ```bash
   curl -X POST http://localhost:8080/perplexity \
     -H "Content-Type: application/json" \
     -d '{
       "text": "The quick brown fox jumps over the lazy dog"
     }'
   ```

### Batch Processing

**Create input file** (`prompts.json`):
```json
[
  "Once upon a time",
  "In a galaxy far away",
  "The little robot"
]
```

**Run batch generation:**
```bash
python inference.py batch \
  --input prompts.json \
  --output results.json \
  --gcs-bucket your-bucket-name \
  --max-length 100 \
  --temperature 0.8
```

## 🚀 Cloud Deployment

### Google Cloud Run

1. **Create Dockerfile:**
```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY slam3_pytorch.py inference.py ./

ENV GCS_BUCKET=your-bucket-name
ENV PORT=8080

CMD exec python inference.py api --gcs-bucket $GCS_BUCKET --port $PORT
```

2. **Build and deploy:**
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/slam3-inference

# Deploy to Cloud Run
gcloud run deploy slam3-inference \
  --image gcr.io/PROJECT_ID/slam3-inference \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --allow-unauthenticated
```

### AWS Lambda

Create `lambda_handler.py`:
```python
from inference import SLAM3Inference
import json

model = None

def handler(event, context):
    global model
    if model is None:
        model = SLAM3Inference(
            gcs_bucket='your-bucket',
            gcs_checkpoint='checkpoints/best.pt'
        )

    body = json.loads(event['body'])
    results = model.generate(
        prompt=body['prompt'],
        max_length=body.get('max_length', 100)
    )

    return {
        'statusCode': 200,
        'body': json.dumps({'generated': results})
    }
```

## 📊 Training Data

### TinyStories Dataset
- **Size**: ~500MB, ~2M stories
- **Domain**: Simple children's stories
- **Vocabulary**: GPT-2 tokenizer (50,257 tokens)
- **Purpose**: Quick training and testing

### Using Custom Data

Create a custom dataset loader:

```python
from datasets import load_dataset
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        # Load your data
        self.data = load_dataset('text', data_files=data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        encoded = self.tokenizer(
            text,
            max_length=self.max_length + 1,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        return input_ids[:-1], input_ids[1:]
```

## 🔧 Troubleshooting

### Out of Memory Errors

**Reduce batch size:**
```python
config.BATCH_SIZE = 8  # or 4
```

**Reduce sequence length:**
```python
config.SEQ_LEN = 64  # or 32
```

**Enable gradient checkpointing:**
```python
config.CHECKPOINT_GRADIENTS = True
```

### Slow Training

**Check GPU usage:**
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

**Increase batch size** (if memory allows):
```python
config.BATCH_SIZE = 32
```

**Use mixed precision training:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    loss = train_step(model, batch, config)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Colab Disconnects

All progress is automatically saved! Just:
1. Reconnect to Colab
2. Run all cells again
3. Training will resume from last checkpoint

## 📈 Monitoring

### Weights & Biases

Enable WandB in the Colab notebook for:
- Real-time loss curves
- Learning rate schedules
- Gradient norms
- Model comparisons

### TensorBoard (Alternative)

Add to training loop:
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/slam3')

# In training loop:
writer.add_scalar('Loss/train', loss, step)
writer.add_scalar('Loss/val', val_loss, step)
writer.add_scalar('Learning Rate', lr, step)
```

View with:
```bash
tensorboard --logdir runs/slam3
```

## 🎯 Performance Tips

### Training
1. **Use larger batch sizes** on better GPUs
2. **Enable mixed precision** for 2x speedup
3. **Adjust warmup steps** based on dataset size
4. **Monitor specialization loss** - should decrease over time
5. **Use gradient accumulation** if memory limited

### Inference
1. **Use best.pt** for production
2. **Reduce EF cycles** during inference for speed
3. **Enable early stopping** (automatic)
4. **Batch multiple prompts** for efficiency
5. **Use lower temperature** for more deterministic outputs

## 📝 Model Statistics

| Parameter | Value |
|-----------|-------|
| Total Parameters | ~15M |
| Embedding Dim | 128 |
| Sequence Length | 128 |
| Number of Experts | 4 |
| Active Experts/Token | 2 |
| EF Cycles | 2 |
| Rotations/Cycle | 4 |
| Total Views/Token | 16 |

## 🔬 Architecture Details

### Specialized Blocks

1. **Block A (Structure)**: Low-frequency patterns
   - Target: [1.0, 0.5, 0.2, 0.1]
   - Focus: Overall structure, long-range dependencies

2. **Block B (Patterns)**: Mid-frequency patterns
   - Target: [0.3, 1.0, 0.8, 0.3]
   - Focus: Recurring patterns, syntax

3. **Block C (Transitions)**: Mid-high frequency
   - Target: [0.1, 0.4, 1.0, 0.7]
   - Focus: State transitions, flow

4. **Block D (Details)**: High-frequency details
   - Target: [0.1, 0.2, 0.5, 1.0]
   - Focus: Fine-grained details, local context

### Loss Components

1. **Cross-Entropy Loss**: Language modeling objective
2. **Load Balance Loss**: Ensures expert utilization
3. **Specialization Loss**: Enforces block specialization

Total loss: `CE + 0.01 * LB + 0.01 * Spec`

## 🤝 Contributing

Contributions welcome! Areas of interest:
- [ ] Flash Attention integration
- [ ] Multi-GPU training support
- [ ] Additional datasets
- [ ] Quantization for deployment
- [ ] Streaming inference
- [ ] Better specialization mechanisms

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Original SLAM3 architecture
- TinyStories dataset by Eldan & Li
- GPT-2 tokenizer by OpenAI
- PyTorch team for the framework

## 📮 Contact

For questions or issues:
- Open an issue on GitHub
- Check existing discussions
- Read the troubleshooting section

## 🔗 Resources

- [Architecture Diagram](read.txt) - Visual explanation
- [Training Notebook](train_colab.ipynb) - Complete training setup
- [Inference Script](inference.py) - Deployment-ready inference
- [Model Code](slam3_pytorch.py) - PyTorch implementation

---

**Status**: ✅ Fully functional | 🚀 Production-ready | 📊 Well-documented

**Last Updated**: 2025-01-13
