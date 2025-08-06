# DCGAN Experiments Log

## Overview
This document tracks experimental results for DCGAN training on celebrity faces dataset.

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