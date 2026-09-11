# Chokkhu: The Sovereign First-Principles Machine Learning & AI Ecosystem
## Master End-to-End Architectural Pipeline & Theoretical Compendium (Milestones 1–20)

---

### Executive Overview & Sovereign Philosophy

`Chokkhu` is an uncompromising, sovereign, zero-heavy-dependency artificial intelligence, machine learning, computer vision, natural language processing, audio, scientific computing, robotics, and generative AI universe built entirely from first mathematical principles in pure `NumPy` and `SciPy`.

- **Strict Zero-Heavy-Dependency Guarantee**: Zero reliance on PyTorch, TensorFlow, JAX, Scikit-Learn, HuggingFace Transformers, SHAP, Librosa, or PyG.
- **Pure Mathematical Transparency**: Every forward pass, backward gradient propagation, loss function, optimizer step, Riemannian manifold operation, and probabilistic sampler is formulated and implemented explicitly.
- **Ultra-Lightweight Distribution**: An entire universe of enterprise-grade AI algorithms packed into a single distribution wheel under **1 MB**.
- **Battle-Tested Resilience**: Complete cross-platform compatibility across Linux, macOS, and Windows with 100% type safety (`mypy` across 311+ source files) and 100% passing test suites.

---

```
                       CHOKKHU ARCHITECTURAL UNIVERSE
===================================================================================
 Layer 1: Core Foundation & Tensor Math (AD, Autograd, Optimizers, Losses, Layers)
-----------------------------------------------------------------------------------
 Layer 2: Classical ML & Tabular Intelligence (Ensembles, Boosters, Trees, SVM, GMM)
-----------------------------------------------------------------------------------
 Layer 3: Deep Representation Learning (CNNs, ViT, Swin, ResNet, NeRF, 3DGS, UNet)
-----------------------------------------------------------------------------------
 Layer 4: Frontier Sequence & LLM Modeling (Transformers, RoPE, MLA, MoE, Mamba, RWKV)
-----------------------------------------------------------------------------------
 Layer 5: Multi-Modal, Audio & Vision-Language (CLIP, SigLIP, Conformer, AudioVAE, Vocoder)
-----------------------------------------------------------------------------------
 Layer 6: Probabilistic & Generative Universe (DDPM, Flow Matching, GFlowNets, VAE)
-----------------------------------------------------------------------------------
 Layer 7: Reinforcement Learning & Multi-Agent (PPO, SAC, Decision Transformer, QMIX)
-----------------------------------------------------------------------------------
 Layer 8: Scientific AI & Physical Systems (PINNs, Neural ODE, FNO, Differentiable Physics)
-----------------------------------------------------------------------------------
 Layer 9: Trustworthy AI, Safety & Geometry (Causal DAGs, Watermarking, HDC, Hyperbolic)
===================================================================================
```

---

## Part I: Completed Subsystems & Architectural Pipelines (Milestones 1–19)

---

### 1. Vector Retrieval & High-Dimensional Search (Milestone 1)
- **Mathematical Mechanics**:
  - Hierarchical Navigable Small World (HNSW) graphs with decaying geometric layer distributions: $P(l) = \exp(-l / m_L)$.
  - Inverted File Product Quantization (IVF-PQ) with sub-vector codebooks and asymmetric distance computation (ADC).
  - Locality-Sensitive Hashing (LSH) using random Gaussian hyperplanes and MinHash Jaccard estimation.
  - Reciprocal Rank Fusion (RRF): $\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$.
- **Module Path**: `src/chokkhu/retrieval/`

---

### 2. Recommendation Systems Universe (Milestone 2)
- **Mathematical Mechanics**:
  - Deep Factorization Machines (DeepFM): $\hat{y} = \langle w, x \rangle + \sum_{i<j} \langle v_i, v_j \rangle x_i x_j + \text{DNN}(x)$.
  - Neural Collaborative Filtering (NCF / NeuMF) fusing GMF and Multi-Layer Perceptrons.
  - Sequential Attention Recommendation (SASRec) with causal multi-head self-attention.
  - Collaborative Metric Learning (CML) and Two-Tower semantic retrieval with in-batch negative sampling.
- **Module Path**: `src/chokkhu/models/recsys/`

---

### 3. Causal Inference & Uplift Modeling (Milestone 3)
- **Mathematical Mechanics**:
  - Double Machine Learning (DML / Robinson's transformation) using cross-fitting.
  - Meta-Learners: S-Learner, T-Learner, X-Learner ($\tau(x) = g(x)\hat{D}_0(x) + (1-g(x))\hat{D}_1(x)$), and Doubly Robust (DR) Estimation.
  - Uplift Random Forests with KL-divergence, Chi-Square, and Euclidean splitting criteria.
  - Cumulative Gain & Qini Curves: $Q = A_{\text{uplift}} - A_{\text{random}}$.
- **Module Path**: `src/chokkhu/causal/`

---

### 4. Survival Analysis & Reliability Engineering (Milestone 4)
- **Mathematical Mechanics**:
  - Kaplan-Meier Non-Parametric Estimator: $\hat{S}(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)$.
  - Nelson-Aalen Cumulative Hazard: $\hat{H}(t) = \sum_{t_i \le t} \frac{d_i}{n_i}$.
  - Cox Proportional Hazards with Breslow & Efron partial likelihood optimization.
  - DeepSurv Neural Hazard Regression: $h(t|x) = h_0(t) \exp(f_\theta(x))$.
  - Concordance Index ($C$-Index) and Integrated Brier Score (IBS).
- **Module Path**: `src/chokkhu/models/survival/`

---

### 5. Multi-Modal Vision-Language Universe (Milestone 5)
- **Mathematical Mechanics**:
  - Contrastive Language-Image Pretraining (CLIP) with symmetric infoNCE loss: $\mathcal{L} = \frac{1}{2} (\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I})$.
  - SigLIP: Sigmoid Cross-Entropy Multi-Modal Learning with learnable temperature $\tau$ and bias $b$.
  - Perceiver Resampler with fixed cross-attention latent queries for arbitrary image feature dimensions.
  - Cross-Attention Multimodal Fusion and Vision-Language Prompt Aligners.
- **Module Path**: `src/chokkhu/models/multimodal/`

---

### 6. Scientific Machine Learning & PINNs (Milestone 6)
- **Mathematical Mechanics**:
  - Physics-Informed Neural Networks (PINNs) with exact autograd/numerical derivative computation for Burgers', Heat, Wave, and Harmonic Oscillator equations:
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{PDE}} \|\mathcal{N}[u](x, t)\|^2 + \lambda_{\text{BC}} \|u - u_{\text{BC}}\|^2$$
  - Neural Ordinary Differential Equations (Neural ODE) with Runge-Kutta 4 (RK4) and Euler numerical integration schemes.
  - Symbolic Genetic Regression for closed-form mathematical law discovery.
- **Module Path**: `src/chokkhu/sciml/`

---

### 7. Algorithmic Fairness & Bias Mitigation (Milestone 7)
- **Mathematical Mechanics**:
  - Metric Suite: Demographic Parity Difference, Equalized Odds, Disparate Impact, Calibration by Group.
  - In-Processing Adversarial Debiasing with gradient reversal layer (GRL) or min-max optimization.
  - Post-Processing Reject Option Classification (ROC) and Equalized Odds Threshold Optimization.
- **Module Path**: `src/chokkhu/fairness/`

---

### 8. Edge AI, Quantization & Model Compression (Milestone 8)
- **Mathematical Mechanics**:
  - INT8/INT4 Uniform Asymmetric & Symmetric Quantization: $q = \text{clip}\left(\text{round}\left(\frac{x}{S}\right) + Z, q_{\min}, q_{\max}\right)$.
  - Structured & Unstructured Magnitude/Gradient Pruning with sparsity tracking.
  - Response & Feature Distillation (Hinton Knowledge Distillation): $\mathcal{L}_{\text{KD}} = (1-\alpha)\mathcal{L}_{\text{CE}} + \alpha \tau^2 \mathcal{L}_{\text{KL}}(\sigma(z_s/\tau), \sigma(z_t/\tau))$.
- **Module Path**: `src/chokkhu/quantization/` & `src/chokkhu/pruning/`

---

### 9. 3D Computer Vision, Point Clouds & NeRF / 3DGS (Milestone 9)
- **Mathematical Mechanics**:
  - PointNet & PointNet++ with Spatial Transformer T-Net, Farthest Point Sampling (FPS), and Ball Query Set Abstraction.
  - Neural Radiance Fields (NeRF) with Positional Fourier Encoding $\gamma(p) = [\sin(2^k \pi p), \cos(2^k \pi p)]$ and Volume Rendering Quadrature.
  - 3D Gaussian Splatting with 3D covariance matrix $\Sigma = R S S^T R^T$ and differentiable point rasterization.
- **Module Path**: `src/chokkhu/models/vision3d/`

---

### 10. Autonomous AI Agents & Execution Engines (Milestone 10)
- **Mathematical Mechanics**:
  - ReAct (Reasoning + Acting) loop with scratchpad observation-action trajectory management.
  - Tree-of-Thoughts (ToT) Breadth-First & Depth-First search with branch evaluation heuristics.
  - Graph-of-Thoughts (GoT) arbitrary DAG thought aggregation and transformation.
  - Reflexion self-correction with episodic memory and dynamic self-consistency voting.
- **Module Path**: `src/chokkhu/models/nlp/agents/`

---

### 11. Reinforcement Learning Universe (Milestone 11)
- **Mathematical Mechanics**:
  - Proximal Policy Optimization (PPO) with clipped surrogate objective: $\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$.
  - Soft Actor-Critic (SAC) with maximum entropy reinforcement learning: $J(\pi) = \mathbb{E} [R(s, a) + \alpha \mathcal{H}(\pi(\cdot|s))]$.
  - Deep Q-Networks (DQN, Double DQN, Dueling DQN) with Prioritized Experience Replay (PER).
  - Decision Transformer: Sequence modeling of autoregressive trajectories $(R_1, s_1, a_1, \dots, R_T, s_T, a_T)$.
- **Module Path**: `src/chokkhu/rl/`

---

### 12. Advanced Graph Neural Networks Universe (Milestone 12)
- **Mathematical Mechanics**:
  - Graph Convolutional Networks (GCN) with symmetric normalized adjacency $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$.
  - Graph Attention Networks (GATv2) with dynamic parameterized LeakyReLU attention coefficients.
  - GraphSAGE with Mean, Pooling, and LSTM neighborhood aggregation.
  - Graph Isomorphism Networks (GIN) with maximal expressive power: $h_v^{(k)} = \text{MLP}^{(k)} \left((1 + \epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}\right)$.
- **Module Path**: `src/chokkhu/models/gnn/`

---

### 13. Frontier LLMs, Linear Attention & Alignment (Milestone 13)
- **Mathematical Mechanics**:
  - Multi-Head Latent Attention (MLA) with low-rank key-value compression and decoupled RoPE.
  - DeepSeek-Style Mixture of Experts (MoE) with Top-$K$ gating, auxiliary load-balancing loss, and shared experts.
  - State Space Models (Mamba / S4) with input-dependent selective discretization: $\bar{A} = \exp(\Delta A), \bar{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot \Delta B$.
  - Linear Attention (FlashLinearAttention, RWKV-style time-mixing recurrence).
  - Direct Preference Optimization (DPO): $\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma \left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$.
- **Module Path**: `src/chokkhu/models/nlp/` & `src/chokkhu/models/llm/`

---

### 14. Next-Gen Generative AI, LoRA, Conformal & Stacking (Milestone 14)
- **Mathematical Mechanics**:
  - Continuous Normalizing Flows (CNF) & Flow Matching with Optimal Transport paths: $\frac{dx_t}{dt} = v_\theta(x_t, t)$.
  - Low-Rank Adaptation (LoRA) parameter-efficient fine-tuning: $W = W_0 + \frac{\alpha}{r} B A$.
  - Conformal Prediction for distribution-free uncertainty intervals: $C(X_{n+1}) = [\hat{y} - \hat{q}_{1-\alpha}, \hat{y} + \hat{q}_{1-\alpha}]$.
  - Out-of-Fold Super Learner & Stacking Regressor/Classifier with meta-estimators.
- **Module Path**: `src/chokkhu/models/generative/` & `src/chokkhu/uncertainty/`

---

### 15. Privacy, Quantum ML, Geo-Spatial, NAS & TDA (Milestone 15)
- **Mathematical Mechanics**:
  - Differential Privacy: DP-SGD with per-sample gradient clipping and Gaussian noise perturbation.
  - Parameterized Quantum Circuits (PQC) & Quantum Neural Networks with Pauli-X/Y/Z rotation gates.
  - Geo-Spatial Spatio-Temporal Kriging, Haversine spatial embeddings, and ST-Conv layers.
  - Neural Architecture Search (Differentiable DARTS & Genetic Supernet NAS).
  - Topological Data Analysis (TDA) Vietoris-Rips filtration, Persistent Homology, and Betti numbers.
- **Module Path**: `src/chokkhu/privacy/`, `src/chokkhu/models/quantum/`, `src/chokkhu/geospatial/`, `src/chokkhu/tda/`

---

### 16. Spiking Neural Networks, Equilibrium Models & GFlowNets (Milestone 16)
- **Mathematical Mechanics**:
  - Leaky Integrate-and-Fire (LIF) Spiking Neural Networks with surrogate gradient backpropagation:
    $$V[t] = \beta V[t-1] + I[t] - S[t-1] V_{\text{reset}}, \quad S[t] = \Theta(V[t] - V_{\text{th}})$$
  - Deep Equilibrium Models (DEQ) solving fixed points $z^* = f_\theta(z^*, x)$ with implicit function theorem backward passes.
  - Generative Flow Networks (GFlowNets) with Trajectory Balance (TB) and Detailed Balance (DB) objectives.
- **Module Path**: `src/chokkhu/models/neuromorphic/`, `src/chokkhu/models/deq/`, `src/chokkhu/models/gflownet/`

---

### 17. Production Generative AI, Audio Codecs & Genetic AutoML (Milestone 17)
- **Mathematical Mechanics**:
  - EnCodec / SoundStream Neural Audio Compression: Strided convolutions + Residual Vector Quantization (RVQ) with codebook EMA.
  - Speculative Decoding Engine: Draft model verification with speculative acceptance criteria.
  - Fourier Neural Operators (FNO) for continuous PDE solving across arbitrary mesh resolutions:
    $$u(x) = \mathcal{F}^{-1} \left( R \cdot \mathcal{F}(u) \right)(x) + W u(x)$$
  - Genetic AutoML Pipeline discovering optimal preprocessing, feature engineering, and ensemble graphs.
- **Module Path**: `src/chokkhu/models/audio/codec.py`, `src/chokkhu/models/llm/speculative.py`, `src/chokkhu/sciml/fno.py`, `src/chokkhu/automl/genetic_pipeline.py`

---

### 18. Edge Serving, Hybrid State-Space & Optimal Transport (Milestone 18)
- **Mathematical Mechanics**:
  - ONNX / TensorRT-compatible C-Header & Pure Python Zero-Dependency Runtime Exporter.
  - Jamba-Style Hybrid Transformer-SSM Block: Parallel Mamba selective recurrence + MLA Multi-Head Latent Attention.
  - Continual Learning: Elastic Weight Consolidation (EWC) using empirical Fisher Information Matrix:
    $$\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_{A, i}^*)^2$$
  - Sinkhorn Optimal Transport with entropy regularization: $P^* = \text{diag}(u) K \text{diag}(v)$ where $K_{ij} = \exp(-C_{ij}/\varepsilon)$.
- **Module Path**: `src/chokkhu/serving/`, `src/chokkhu/models/hybrid/`, `src/chokkhu/continual/`, `src/chokkhu/optimal_transport/`

---

### 19. Robotics, World Models, Multi-Agent RL, Genomic AI & HDC (Milestone 19)
- **Mathematical Mechanics**:
  - Diffusion Policy: Denoising diffusion probabilistic model (DDPM) parameterized over robot action trajectories conditioned on visual/proprioceptive states.
  - Recurrent World Model: Latent state-space dynamic model $s_{t+1} \sim p(s_{t+1} | s_t, a_t)$ + Model Predictive Path Integral (MPPI) trajectory optimization.
  - QMIX Multi-Agent RL: Non-linear monotonic value factorization $\frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0$ enforced via state-conditioned hypernetworks with absolute weights.
  - GenomicBERT & Protein Direct Coupling Analysis (DCA): 1D DNA $k$-mer transformer with variant effect zero-shot scoring and DCA residue contact maps via inverse covariance matrix.
  - Continuous Fuzzy Neuro-Symbolic Logic & Knowledge Graph Embeddings (RotatE, TransE).
  - Hyperdimensional Computing (HDC): High-dimensional vector symbolic algebra (Binding $\odot$, Bundling $\sum$, Permutation $\Pi$) with zero-gradient associative memory classification.
- **Module Path**: `src/chokkhu/models/robotics/`, `src/chokkhu/models/marl/`, `src/chokkhu/models/bio/`, `src/chokkhu/models/neurosymbolic/`, `src/chokkhu/hdc/`

---

## Part II: The Next Frontier — Milestone 20 Architecture & Technical Specification

Milestone 20 unlocks advanced frontier domains in AI safety, discrete sequence diffusion, robotics mechanics, causal DAG discovery, and non-Euclidean hyperbolic manifolds.

```
+-----------------------------------------------------------------------------------------------+
|                                MILESTONE 20 ARCHITECTURE MATRIX                               |
+--------------------+-------------------------------------------+------------------------------+
| Subsystem          | Components / Classes                      | Core Mathematical Foundation |
+--------------------+-------------------------------------------+------------------------------+
| 20.1 Discrete Text | ScoreEntropyDiscreteDiffusion             | Discrete Markov Transitions  |
|      Diffusion     | DiscreteTextDiffusion                     | $q(x_t|x_0)$, Absorbing Mask |
+--------------------+-------------------------------------------+------------------------------+
| 20.2 Differentiable| RobotArmKinematics,                       | Denavit-Hartenberg (DH),     |
|      Physics       | ForwardInverseKinematics, Damped Least Sq | Jacobian DLS Inverse, SPH    |
|      & Robotics    | DifferentiableParticleFluid               | Particle Navier-Stokes       |
+--------------------+-------------------------------------------+------------------------------+
| 20.3 AI Safety &   | StatisticalTextWatermark                  | Green/Red Hash Partitioning, |
|      Watermarking  | RefusalDirectionProbe                     | Z-Score Test, Latent Probe   |
+--------------------+-------------------------------------------+------------------------------+
| 20.4 Causal DAG    | NOTEARSCausalDiscovery                    | Matrix Exponential Trace     |
|      Discovery     | PCAlgorithm                               | $\text{tr}(e^W)-d=0$, CI Test|
+--------------------+-------------------------------------------+------------------------------+
| 20.5 Hyperbolic    | PoincareBallEmbedding                     | Riemannian Metric Tensor,    |
|      Geometry      | LorentzManifold                           | Möbius Addition, Minkowski   |
+--------------------+-------------------------------------------+------------------------------+
```

---

### Detailed Mathematical Blueprints for Milestone 20

#### 20.1 Discrete Text Diffusion (`src/chokkhu/models/generative/discrete_diffusion.py`)
- **Mathematical Principle**: Unlike continuous Gaussian diffusion, discrete diffusion operates over categorical token spaces $\{1, \dots, V\}$. The forward Markov process corrupts tokens via a transition matrix $Q_t$:
  $$q(x_t | x_{t-1}) = \text{Cat}(x_t; x_{t-1} Q_t)$$
  Using an absorbing state $[MASK]$, the marginal distribution is $q(x_t | x_0) = \alpha_t x_0 + (1 - \alpha_t) [MASK]$.
- **Training Objective**: Categorical cross-entropy / ELBO variational lower bound optimizing token reconstruction from unmasked context.

#### 20.2 Differentiable Physics & Robotics Kinematics (`src/chokkhu/models/robotics/kinematics.py`, `src/chokkhu/sciml/fluid.py`)
- **Forward Kinematics (DH Convention)**: Homogeneous transformation matrices:
  $$T_i^{i-1} = \begin{bmatrix} \cos\theta_i & -\sin\theta_i \cos\alpha_i & \sin\theta_i \sin\alpha_i & a_i \cos\theta_i \\ \sin\theta_i & \cos\theta_i \cos\alpha_i & -\cos\theta_i \sin\alpha_i & a_i \sin\theta_i \\ 0 & \sin\alpha_i & \cos\alpha_i & d_i \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
- **Inverse Kinematics (Damped Least Squares / Levenberg-Marquardt)**:
  $$\Delta \theta = J^T (J J^T + \lambda^2 I)^{-1} e$$
- **Smoothed Particle Hydrodynamics (SPH)**: Density estimation $\rho_i = \sum_j m_j W(\|r_i - r_j\|, h)$ and Navier-Stokes pressure/viscosity forces.

#### 20.3 AI Safety & LLM Watermarking (`src/chokkhu/safety/`)
- **Kirchenbauer Statistical Watermark**: Deterministically seeds a pseudo-random hash generator with prefix token $x_{t-1}$ to partition the vocabulary into Green list $G$ (size $\gamma V$) and Red list $R$. Adds a logit bias $\delta$ to Green tokens during generation. Detection computes:
  $$z = \frac{|x|_G - \gamma T}{\sqrt{T \gamma (1 - \gamma)}}$$
  Rejecting unwatermarked hypothesis when $z > z_{\text{threshold}}$.
- **Refusal Direction Probe**: Computes mean difference direction $v = \mu_{\text{refusal}} - \mu_{\text{compliant}}$ in transformer residual activations, allowing directional steering or safety monitoring without retraining.

#### 20.4 Causal Discovery DAGs (`src/chokkhu/causal/discovery.py`)
- **NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented Lagrangian for Structure learning)**:
  $$\min_{W} \frac{1}{2n} \|X - X W\|_F^2 + \lambda \|W\|_1 \quad \text{s.t.} \quad h(W) = \text{tr}(e^{W \odot W}) - d = 0$$
- **PC Algorithm**: Constraint-based causal discovery utilizing partial correlation and Fisher $z$-transformation conditional independence tests to discover the Markov equivalence class (CPDAG).

#### 20.5 Hyperbolic Manifold Geometry (`src/chokkhu/geometry/hyperbolic.py`)
- **Poincaré Ball Model**: $d$-dimensional open ball $\mathbb{D}^d = \{x \in \mathbb{R}^d : \|x\| < 1\}$ equipped with Riemannian metric $g_x = \left(\frac{2}{1 - \|x\|^2}\right)^2 I$.
- **Geodesic Distance**: $d_{\mathbb{D}}(u, v) = \text{arcosh}\left(1 + 2 \frac{\|u - v\|^2}{(1 - \|u\|^2)(1 - \|v\|^2)}\right)$.
- **Möbius Addition**: $u \oplus v = \frac{(1 + 2\langle u, v \rangle + \|v\|^2)u + (1 - \|u\|^2)v}{1 + 2\langle u, v \rangle + \|u\|^2 \|v\|^2}$.
- **Lorentz (Hyperboloid) Model**: Minkowski inner product $\langle x, y \rangle_L = -x_0 y_0 + \sum_{i=1}^d x_i y_i$ enabling high-capacity representation of scale-free hierarchical tree networks without distortion.

---

### Verification and Delivery Standards

1. **Exact Mathematical Consistency**: Every formula tested against analytical ground truths.
2. **Deterministic Reproducibility**: Explicit random seed parameterization across all probabilistic modules.
3. **Zero External Runtime Heavy Dependencies**: Strictly pure NumPy and SciPy.
4. **Complete Documentation & Type Coverage**: Exhaustive docstrings with asymptotic complexity notes, 100% mypy pass rate.
