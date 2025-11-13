"""
Quick test script to verify SLAM3 PyTorch implementation
Run this locally before uploading to Colab
"""

import torch
from slam3_pytorch import SLAM3Model, SLAM3Config
import time

def test_model():
    print("\n" + "="*80)
    print("🧪 SLAM3 PyTorch Quick Test")
    print("="*80 + "\n")

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Initialize model
    print("\n✓ Initializing model...")
    config = SLAM3Config()
    model = SLAM3Model(config)
    model = model.to(device)

    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Embed Dim: {config.EMBED_DIM}")
    print(f"  Sequence Length: {config.SEQ_LEN}")
    print(f"  EF Cycles: {config.MAX_EF_CYCLES}")

    # Test forward pass
    print("\n✓ Testing forward pass...")
    batch_size = 2
    seq_len = config.SEQ_LEN
    tokens = torch.randint(0, config.VOCAB_SIZE, (batch_size, seq_len), device=device)

    start_time = time.time()
    with torch.no_grad():
        output = model(tokens, training=False)
    forward_time = time.time() - start_time

    print(f"  Input shape: {tokens.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Forward pass time: {forward_time*1000:.1f}ms")
    print(f"  Tokens/second: {(batch_size * seq_len / forward_time):.0f}")

    # Test backward pass
    print("\n✓ Testing backward pass...")
    model.train()
    tokens = torch.randint(0, config.VOCAB_SIZE, (batch_size, seq_len), device=device)
    targets = torch.randint(0, config.VOCAB_SIZE, (batch_size, seq_len), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    start_time = time.time()
    optimizer.zero_grad()
    logits = model(tokens, training=True)
    logits_flat = logits.reshape(-1, config.VOCAB_SIZE)
    targets_flat = targets.reshape(-1)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    loss.backward()
    optimizer.step()
    backward_time = time.time() - start_time

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Backward pass time: {backward_time*1000:.1f}ms")
    print(f"  Full step time: {backward_time*1000:.1f}ms")

    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"\n✓ GPU Memory:")
        print(f"  Allocated: {memory_allocated:.2f} GB")
        print(f"  Reserved: {memory_reserved:.2f} GB")

    # Test specialization
    print("\n✓ Testing specialization mechanism...")
    spec_loss = model.encoder.get_specialization_loss()
    lb_loss = model.encoder.get_load_balance_loss()
    print(f"  Specialization loss: {spec_loss:.4f}")
    print(f"  Load balance loss: {lb_loss:.4f}")

    # Test all cycles
    print("\n✓ Testing all EF cycles...")
    for i, cycle in enumerate(model.encoder.ef_cycles):
        print(f"  Cycle {i+1}:")
        for j, block in enumerate(cycle.blocks):
            print(f"    Block {j} ({block.block_name}): ", end="")
            if block.last_freq_profile is not None:
                profile = block.last_freq_profile.detach().cpu().numpy()
                print(f"Profile = [{', '.join(f'{x:.2f}' for x in profile)}]")
            else:
                print("Not computed yet")

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nYour SLAM3 model is ready for training!")
    print("\nNext steps:")
    print("1. Upload files to your GitHub repository")
    print("2. Open train_colab.ipynb in Google Colab")
    print("3. Follow the notebook instructions")
    print("4. Start training!")
    print("\n" + "="*80 + "\n")

    return True


if __name__ == '__main__':
    try:
        success = test_model()
        if success:
            exit(0)
        else:
            exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
