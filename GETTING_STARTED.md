# 🚀 Getting Started with SLAM3 Training

This guide will help you start training SLAM3 on Google Colab in under 10 minutes!

## 📋 Prerequisites

Before starting, make sure you have:

1. ✅ **Google Account** - For Google Colab and Cloud Storage
2. ✅ **GitHub Account** - For version control
3. ✅ **Google Cloud Project** - Create one at [console.cloud.google.com](https://console.cloud.google.com)

## 🎯 Quick Start (5 Steps)

### Step 1: Test Locally (Optional but Recommended)

```bash
cd /Users/vishsiddharth/Desktop/NEWSLAM

# Install dependencies
pip install torch transformers

# Run quick test
python quick_test.py
```

This will verify the model works on your machine. Expected output:
```
✅ ALL TESTS PASSED!
Your SLAM3 model is ready for training!
```

### Step 2: Create GitHub Repository

Two options:

**Option A: Using GitHub Web Interface**
1. Go to [github.com/new](https://github.com/new)
2. Repository name: `slam3-training`
3. Make it **Public** (or Private if you prefer)
4. Click "Create repository"

**Option B: Using GitHub CLI** (if you have `gh` installed)
```bash
cd /Users/vishsiddharth/Desktop/NEWSLAM
gh repo create slam3-training --public --description "SLAM3 Model Training"
```

### Step 3: Push Code to GitHub

```bash
cd /Users/vishsiddharth/Desktop/NEWSLAM

# Initialize git
git init
git add .
git commit -m "Initial commit: SLAM3 training setup"

# Add your remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/slam3-training.git
git branch -M main
git push -u origin main
```

### Step 4: Setup Google Cloud Project

1. **Create GCP Project** (if you don't have one)
   - Go to [console.cloud.google.com](https://console.cloud.google.com)
   - Click "Create Project"
   - Name it: `slam3-training`
   - Note your Project ID

2. **Enable Required APIs**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable storage-api.googleapis.com
   ```

3. **Create GitHub Personal Access Token**
   - Go to [github.com/settings/tokens](https://github.com/settings/tokens)
   - Click "Generate new token (classic)"
   - Name: `slam3-colab`
   - Scopes: Select `repo` and `workflow`
   - Click "Generate token"
   - **Copy the token** - you'll need it in Colab!

### Step 5: Open Colab and Start Training!

1. **Upload Notebook to Colab**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - File → Upload notebook
   - Select `train_colab.ipynb` from your computer

2. **Change Runtime to GPU**
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU** (or TPU if available)
   - Click Save

3. **Run All Cells**
   - Runtime → Run all
   - Or click the play button on each cell

4. **Follow Prompts**
   The notebook will ask you for:
   - Google Cloud Project ID
   - GitHub token (paste the one you created)
   - Bucket name (can create new or use existing)
   - Git email and name

5. **Training Starts!**
   - TinyStories dataset downloads (~500MB)
   - Training begins automatically
   - Checkpoints saved every 500 steps
   - Can disconnect anytime - progress saved!

## 📊 What to Expect

### Training Progress

```
Step 10/50000 | Loss: 8.3421 | CE: 8.3215 | LR: 6.00e-06 | 450ms/step
Step 20/50000 | Loss: 7.8234 | CE: 7.8012 | LR: 1.20e-05 | 445ms/step
...

============================================================
📊 Evaluation at step 100
   Val Loss: 6.8432
============================================================

💾 Saving checkpoint at step 500...
   ✅ Saved to: gs://slam3-training-abc123/checkpoints/checkpoint_step_500.pt
```

### Timeline

| Milestone | Steps | Time (T4 GPU) | Loss (approx) |
|-----------|-------|---------------|---------------|
| Start | 0 | 0min | ~10.0 |
| First improvement | 100 | ~5min | ~6.0 |
| First checkpoint | 500 | ~25min | ~4.5 |
| Good quality | 5000 | ~4hrs | ~2.5 |
| Publication quality | 50000 | ~40hrs | ~1.5 |

### What Gets Saved

1. **Google Cloud Storage**
   - `checkpoints/checkpoint_step_500.pt`
   - `checkpoints/checkpoint_step_1000.pt`
   - `checkpoints/latest.pt` (always points to most recent)
   - `checkpoints/best.pt` (lowest validation loss)

2. **GitHub Repository**
   - `training_log.json` (updated every checkpoint)
   - Contains: step number, loss, timestamp, GCS path

3. **Weights & Biases** (if enabled)
   - Real-time loss curves
   - Learning rate schedule
   - GPU utilization
   - All hyperparameters

## 🎓 Understanding the Training

### What is SLAM3 Learning?

The model learns to predict the next word in stories. For example:

**Input:** "Once upon a time, there was a little"
**Target:** "girl" (or "boy", "cat", etc.)

### Training Dataset: TinyStories

- 2M simple children's stories
- Vocabulary: 50,257 words (GPT-2)
- Average story length: ~100 words
- Perfect for testing language model architectures

### Loss Metrics

1. **Cross-Entropy Loss** (main metric)
   - Measures prediction accuracy
   - Lower = better
   - Target: < 2.0 for good quality

2. **Load Balance Loss**
   - Ensures all experts are used
   - Should decrease and stabilize
   - Target: < 0.1

3. **Specialization Loss**
   - Encourages blocks to focus on different patterns
   - Should decrease over time
   - Target: < 0.5

## 🛑 Stopping and Resuming

### Graceful Stop

Press **Ctrl+C** or click "Stop" in Colab:
```
⚠️  Training interrupted by user
💾 Saving final checkpoint...
   ✅ Saved to: gs://your-bucket/checkpoints/...

✅ TRAINING COMPLETE!
```

### Resume Training

Just run the notebook again! It will:
1. Detect existing checkpoint in GCS
2. Load model state, optimizer state, metrics
3. Continue from where you left off

```
📥 Loading checkpoint from GCS...
   ✅ Resumed from step 5000 (loss: 3.2145)
```

## 🔍 Monitoring Training

### In the Notebook

The notebook displays progress every 10 steps:
```
Step 100/50000 | Loss: 6.4321 | CE: 6.4102 | LR: 3.00e-04 | 442ms/step
```

### Weights & Biases (Optional)

If you enabled WandB:
1. Go to [wandb.ai](https://wandb.ai)
2. Find your project: `slam3-training`
3. View real-time charts

### Google Cloud Console

Check your checkpoints:
1. Go to [console.cloud.google.com/storage](https://console.cloud.google.com/storage)
2. Click your bucket
3. Navigate to `checkpoints/`

## 🧪 Testing Your Model

### During Training

The notebook includes a generation test at the end. Example output:
```
Prompt: Once upon a time, there was a
Generated: Once upon a time, there was a little girl named Lily.
She liked to play in the park with her friends...
```

### After Training

Download your best checkpoint and test locally:

```bash
# Download checkpoint
gsutil cp gs://your-bucket/checkpoints/best.pt ./

# Test generation
python inference.py \
  --checkpoint best.pt \
  --prompt "Once upon a time" \
  --max-length 100
```

## 💡 Tips for Success

### 1. Start Small
- First run: 1000 steps (~40 minutes)
- Verify everything works
- Then run full 50,000 steps

### 2. Monitor Early
- Check first 100 steps
- Loss should decrease from ~10 to ~6
- If it doesn't, something is wrong

### 3. Use Checkpoints
- Don't worry about disconnects
- Everything auto-saves
- Resume anytime

### 4. Experiment
- Try different temperatures (0.5 - 1.0)
- Adjust learning rate
- Change model size

### 5. Share Progress
- GitHub tracks your training
- Share repository with collaborators
- Show generations to get feedback

## ❓ Common Issues

### "Out of Memory" Error

**Solution 1:** Reduce batch size
```python
config.BATCH_SIZE = 8  # or 4
```

**Solution 2:** Reduce sequence length
```python
config.SEQ_LEN = 64  # instead of 128
```

### "Cannot connect to GitHub"

Check your token:
- Has `repo` and `workflow` permissions
- Not expired
- Copied correctly (no extra spaces)

### "GCS bucket not found"

Make sure:
- Project ID is correct
- Bucket name is correct
- You're authenticated to GCP

### Training is Slow

Normal speeds:
- **CPU**: ~5-10 seconds/step (not recommended)
- **T4 GPU**: ~400-500ms/step (good)
- **V100 GPU**: ~200-300ms/step (great)
- **TPU**: ~100-200ms/step (excellent)

If slower:
- Check runtime type in Colab
- Make sure GPU is enabled
- Restart runtime

## 🎉 Next Steps

Once training completes:

1. **Test Generation**
   ```python
   python inference.py --checkpoint best.pt --prompt "Your prompt here"
   ```

2. **Deploy API**
   ```bash
   python inference.py api --checkpoint best.pt --port 8080
   ```

3. **Share Your Model**
   - Upload to Hugging Face Hub
   - Write a blog post
   - Share on GitHub

4. **Improve Architecture**
   - Try different hyperparameters
   - Add more training data
   - Experiment with specialization

## 📚 Additional Resources

- **Full Documentation**: [README.md](README.md)
- **Model Architecture**: [read.txt](read.txt)
- **PyTorch Code**: [slam3_pytorch.py](slam3_pytorch.py)
- **Inference API**: [inference.py](inference.py)

## 🆘 Need Help?

1. Check [README.md](README.md) troubleshooting section
2. Open an issue on GitHub
3. Review the notebook comments
4. Read error messages carefully

## ✅ Checklist

Before starting:
- [ ] Tested model locally (`python quick_test.py`)
- [ ] Created GitHub repository
- [ ] Pushed code to GitHub
- [ ] Have Google Cloud Project ID
- [ ] Created GitHub token
- [ ] Uploaded notebook to Colab
- [ ] Changed runtime to GPU
- [ ] Ready to train!

---

**Good luck with your training! 🚀**

Questions? Open an issue or check the [README.md](README.md)
