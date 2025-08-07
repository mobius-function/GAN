# DCGAN Experiments Log

## Overview
This document tracks experimental results for DCGAN training on celebrity faces dataset.

---

## Architecture Comparison: Current vs Original DCGAN Paper

### Generator Architecture

| Component | Original DCGAN Paper | Current Implementation | Differences |
|-----------|---------------------|------------------------|-------------|
| Initial Layer | Transposed Conv (4×4, stride=1, no padding) | Linear → Reshape | Uses Linear layer instead of TransposedConv |
| Upsampling Method | TransposedConv2d (stride=2) | Upsample + Conv2d | Bilinear upsampling + regular convolution |
| Activation Function | ReLU | LeakyReLU(0.2) | LeakyReLU instead of ReLU |
| Final Activation | Tanh | Tanh | ✓ Same |
| Batch Normalization | All layers except output | All layers with momentum=0.8 | Custom momentum value (0.8 vs default 0.9) |
| Kernel Size | 4×4 | 3×3 | Smaller kernels |
| Architecture Flow | z → project → reshape → 4 TransposedConv blocks | z → Linear → reshape → 4 Upsample+Conv blocks | Different upsampling strategy |

### Discriminator Architecture

| Component | Original DCGAN Paper | Current Implementation | Differences |
|-----------|---------------------|------------------------|-------------|
| Downsampling Method | Conv2d (stride=2) | Conv2d (stride=2) | ✓ Same approach |
| Activation Function | LeakyReLU(0.2) | LeakyReLU(0.2) | ✓ Same |
| Batch Normalization | All layers except input | All layers except input (momentum=0.8) | Custom momentum value |
| Dropout | None | Dropout2d(0.25) after each block | Added dropout for regularization |
| Final Layer | Linear → no activation | Linear → Sigmoid | Added Sigmoid activation |
| Kernel Size | 4×4 | 3×3 | Smaller kernels |
| Architecture Flow | 4 Conv blocks → flatten → Linear | 4 Conv blocks → flatten → Linear → Sigmoid | Added output activation |

### Training Configuration

| Component | Original DCGAN Paper | Current Implementation | Notes |
|-----------|---------------------|------------------------|--------|
| Loss Function | Binary Cross Entropy | BCE (to be verified) | Check train.py |
| Optimizer | Adam(lr=0.0002, β1=0.5, β2=0.999) | Adam (config dependent) | Check config_celebrity.yaml |
| Weight Initialization | Normal(0, 0.02) | Not specified in models.py | May use PyTorch defaults |
| Input Normalization | [-1, 1] | Assumed [-1, 1] (Tanh output) | Matches generator output range |

### Key Architectural Differences Summary:
1. **Generator uses Upsample+Conv instead of TransposedConv** - May lead to fewer checkerboard artifacts
2. **Generator uses LeakyReLU instead of ReLU** - Can help with gradient flow
3. **Smaller kernel sizes (3×3 vs 4×4)** - Different receptive field characteristics  
4. **Added Dropout in Discriminator** - Regularization to prevent overfitting
5. **Sigmoid in Discriminator output** - Explicit probability output
6. **Custom BatchNorm momentum (0.8)** - Faster adaptation during training

---

## Results Table

| Exp # | Date | Config | Batch Size | LR (G/D) | Latent Dim | Epochs | Final G Loss | Final D Loss | FID Score | Training Time | Hardware | Sample Results | Notes |
|-------|------|--------|------------|----------|------------|---------|--------------|--------------|-----------|---------------|----------|----------------|-------|
| 1 | - | baseline | 128 | 0.0002/0.0002 | 100 | 200 | - | - | - | - | - | [epoch_50](samples_celebrity/epoch_050.png), [epoch_100](samples_celebrity/epoch_100.png), [epoch_200](samples_celebrity/epoch_200.png) | Baseline run |
| 2 | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 3 | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 4 | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 5 | - | - | - | - | - | - | - | - | - | - | - | - | - |

---

## Detailed Experiment Notes

### Experiment 1: Baseline
**Config File**: `config_celebrity.yaml`
- Standard DCGAN architecture
- Adam optimizer with beta1=0.5, beta2=0.999
- Generator channels: [512, 256, 128, 64]
- Discriminator channels: [64, 128, 256, 512]

**Observations**:
- [Training stability notes]
- [Mode collapse observations]
- [Quality assessment]

---

### Experiment 2: [Name]
**Changes from baseline**:
- [List changes]

**Observations**:
- [Notes]

---

## Best Configuration Found
**Experiment #**: -  
**Key Parameters**:
- Batch Size: -
- Learning Rate: -
- Architecture: -

**Why it worked**:
- [Reasoning]

## Best Practices Discovered
- [Any insights about hyperparameters]
- [Training tricks that worked]
- [Architecture modifications that helped]

## TODO
- [ ] Implement FID score calculation
- [ ] Try progressive growing
- [ ] Experiment with different loss functions
- [ ] Test with larger image sizes