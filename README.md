# Foundations of Generative AI in Computer Vision

> **Module 3 — Computer Vision & Deep Learning**  
> Academic course notes on Generative AI fundamentals, GAN architecture, and training dynamics.

---

## Table of Contents

- [What is Generative AI?](#what-is-generative-ai)
- [Examples of Generative AI](#examples-of-generative-ai)
- [Why Use Generative AI?](#why-use-generative-ai)
- [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
  - [Core Concept](#core-concept)
  - [Intuitive Explanation](#intuitive-explanation)
  - [GAN Architecture](#gan-architecture)
  - [Training Workflow](#training-workflow)
  - [Mathematical Objective](#mathematical-objective)
- [Limitations of GANs](#limitations-of-gans)
- [Summary](#summary)
- [References](#references)

---

## What is Generative AI?

**Generative AI (GenAI)** in Computer Vision refers to models capable of **generating new images**, rather than only analyzing or classifying existing ones. These models learn the underlying statistical structure of visual data and use it to synthesize novel, realistic content.

> **Analogy — The Art Student**
> 
> Think of a student learning to paint:
> - They observe thousands of existing paintings
> - They internalize patterns, textures, styles, and structures
> - They then produce **entirely new works** that never existed before
>
> Generative AI works the same way: it **learns from data** and produces new, realistic content.

---

## Examples of Generative AI

| Application | Description | Model Type |
|---|---|---|
| AI-generated human faces | Photorealistic faces of non-existent people | StyleGAN, ProGAN |
| Image-to-image translation | Convert sketch → photo, day → night | Pix2Pix, CycleGAN |
| Super-resolution | Enhance low-res images to high-res | SRGAN, ESRGAN |
| Deepfake generation | Realistic face-swapping in video/images | DeepFaceLab, FaceSwap |
| Text-to-image synthesis | Generate images from text prompts | DALL·E, Stable Diffusion |

> **[Image 1 — Required]**  
> Search: `AI generated faces grid GAN StyleGAN`  
> Add here a grid of AI-generated faces (e.g. from thispersondoesnotexist.com)  
> _Purpose: demonstrate immediately the visual power of GenAI_

---

## Why Use Generative AI?

### 1. Data Augmentation
Generating synthetic labeled data to supplement limited or expensive real datasets. Particularly useful in medical imaging, where annotated data is scarce.

### 2. Content Creation
Automating asset generation for video games, digital art, advertising, and entertainment industries — reducing production time and cost.

### 3. Simulation & Training Environments
Creating synthetic virtual environments to train models (e.g., autonomous driving, robotics) before real-world deployment. Provides controlled, diverse, and safe training data.

### 4. Research & Innovation
Applied in healthcare (generating MRI/CT scans), security (adversarial robustness testing), and scientific discovery (protein structure generation, material design).

---

## Generative Adversarial Networks (GANs)

GANs (Goodfellow et al., 2014) are the foundational architecture for image generation. Their core innovation is the **adversarial training framework**: two networks compete, and competition drives quality.

### Core Concept

A GAN is composed of **two competing neural networks** trained simultaneously:

| Network | Symbol | Role |
|---|---|---|
| **Generator** | G | Creates fake images from random noise |
| **Discriminator** | D | Classifies images as real or fake |

> These two networks are trained **jointly** in a competitive (adversarial) process.

---

### Intuitive Explanation

Think of a GAN as a game between two agents:

| Agent | Analogy | Objective |
|---|---|---|
| Generator | A **forger** | Produce fake images so convincing that the detective cannot tell |
| Discriminator | A **detective** | Identify which images are fake and which are real |

Over time:
- The **forger** improves its technique → more realistic fakes
- The **detective** improves its analysis → better at spotting fakes

**Result:** After sufficient training, the generator produces images virtually indistinguishable from real ones.

---

### GAN Architecture

> **[Image 2 — Critical]**  
> Search: `GAN architecture diagram generator discriminator`  
> Must show: Generator → Fake Image → Discriminator ← Real Images → Real/Fake prediction  
> _This diagram is essential to understanding the training loop_

**Workflow diagram:**

```
Random Noise (z) ──→ [ Generator G ] ──→ Fake Image G(z) ──→ ┐
                                                               ├──→ [ Discriminator D ] ──→ Real / Fake
                          Real Images (x ~ p_data) ───────────┘
```

**Component roles:**

```
┌─────────────────────────────────────────────────────────────┐
│  GENERATOR G                                                │
│  Input  : Random noise vector z ~ N(0, I)                   │
│  Output : Synthetic image G(z) ∈ R^(H×W×C)                 │
│  Goal   : Minimize D's ability to detect fakes              │
├─────────────────────────────────────────────────────────────┤
│  DISCRIMINATOR D                                            │
│  Input  : An image (real or fake)                           │
│  Output : Probability P(image is real) ∈ [0, 1]            │
│  Goal   : Correctly classify real vs. generated images      │
└─────────────────────────────────────────────────────────────┘
```

---

### Training Workflow

> **[Image 3 — Strongly Recommended]**  
> Search: `GAN training progression images epochs`  
> Shows: blurry/noisy images at epoch 1 → sharp, realistic images at convergence  
> _Demonstrates that you understand the iterative learning process_

**Step-by-step training loop:**

1. **Sample noise** — Draw a random latent vector $z \sim p_z(z)$ (e.g., Gaussian distribution)
2. **Generate fake image** — Pass $z$ through the Generator to obtain $G(z)$
3. **Discriminator evaluation** — D receives both real images $x \sim p_{data}$ and fake images $G(z)$; outputs a probability for each
4. **Compute losses** — Calculate the adversarial loss for both D and G
5. **Backpropagation** — Update D's weights to improve classification, then update G's weights to better fool D
6. **Repeat** — Iterate until D outputs ~0.5 for all inputs (Nash equilibrium: D can no longer distinguish real from fake)

> **[Image 4 — Optional but Impactful]**  
> Search: `deepfake example before after`  
> _Connects theory to real-world applications and raises ethical awareness_

---

### Mathematical Objective

The GAN training objective is a **minimax two-player game**:

$$\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**Interpretation of each term:**

| Term | Meaning |
|---|---|
| $\mathbb{E}_{x \sim p_{data}}[\log D(x)]$ | D should output 1 (real) for real images — maximized when D correctly identifies real data |
| $\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$ | D should output 0 (fake) for generated images — maximized when D correctly rejects fakes |

**Each network's goal:**

```
Discriminator D  →  MAXIMIZE V(D, G)   (better classification)
Generator G      →  MINIMIZE V(D, G)   (fool the discriminator)
```

At Nash equilibrium, $p_G = p_{data}$ and $D(x) = \frac{1}{2}$ for all $x$.

---

## Limitations of GANs

Despite their power, GANs come with significant training challenges:

### Training Instability
The minimax game can diverge if G and D are not balanced. If D becomes too strong, G receives no useful gradient signal (vanishing gradients). If G becomes too strong, D collapses.

**Mitigation:** Use Wasserstein GAN (WGAN), gradient penalty (WGAN-GP), or spectral normalization.

### Mode Collapse
The generator learns to produce only a limited variety of outputs, ignoring the full diversity of the training distribution. G "cheats" by repeatedly producing the same few samples that fool D.

**Mitigation:** Mini-batch discrimination, unrolled GANs, or diversity-promoting losses.

### Hyperparameter Sensitivity
GANs are notoriously difficult to tune. Small changes in learning rate, batch size, or architecture can completely destabilize training.

**Mitigation:** Use established architectures (DCGAN, StyleGAN2) with validated hyperparameters as starting points.

> GANs are powerful but require careful architecture design and training strategies. These limitations motivated the development of newer generative models such as **Variational Autoencoders (VAEs)** and **Diffusion Models**.

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Generative AI** | Models that *create* new images, not just analyze existing ones |
| **GAN Framework** | Two-network adversarial game: Generator vs. Discriminator |
| **Training Objective** | Minimax game — G minimizes, D maximizes V(D, G) |
| **Nash Equilibrium** | Training converges when D(x) = 0.5 for all inputs |
| **Main Limitations** | Training instability, mode collapse, hyperparameter sensitivity |
| **Modern Successors** | Diffusion Models (DDPM, Stable Diffusion) now surpass GANs in image quality |

> Generative AI represents a fundamental shift in computer vision: machines are no longer just pattern recognizers — they are **creators**.

---

## References

- Goodfellow, I., et al. (2014). *Generative Adversarial Nets*. NeurIPS. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Radford, A., et al. (2015). *Unsupervised Representation Learning with Deep Convolutional GANs*. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- Karras, T., et al. (2019). *A Style-Based Generator Architecture for GANs*. CVPR. [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)
- Isola, P., et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*. CVPR. [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)
- Arjovsky, M., et al. (2017). *Wasserstein GAN*. ICML. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)

---

*Module 3 — Computer Vision · Generative AI Foundations*  
*Last updated: April 2026*
