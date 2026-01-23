
# Algorithm 1: Robust Evolutionary Hyperparameter Optimization for Autoencoder-based Anomaly Detection

**Input:**
*   $\mathcal{D}_{train} = \{(x_i, y_i)\}_{i=1}^N$: Imbalanced Training Dataset, where $x_i \in \mathbb{R}^d$ and $y_i \in \{0, 1\}$.
*   $\Theta$: Hyperparameter Search Space (Layers $L \in [1,6]$, Units $U \in [16, 512]$, Activations $\phi$, Latent Dim $z \in [4, 128]$).
*   $P$: Population Size for Evolutionary Algorithm (e.g., PSO).
*   $T$: Number of Optimization Iterations.

**Output:**
*   $\theta^*$: Optimal Hyperparameter Configuration maximizing AUPRC.

---

### Procedure

1.  **Stratified Data Preparation**:
    *   Construct $\mathcal{D}_{opt} \subset \mathcal{D}_{train}$ by undersampling majority class $y=0$ to size $N_{sample}$.
    *   Construct $\mathcal{D}_{val}$ by performing a stratified split on $\mathcal{D}_{opt}$, ensuring $P(y=1|\mathcal{D}_{val}) \approx P(y=1|\mathcal{D}_{opt})$.
    *   *Constraint*: Evaluation set must reflect the true anomaly prevalence to prevent metric inflation.

2.  **Meta-Heuristic Optimization Loop**:
    *   Initialize population $\mathcal{P} = \{\theta_1, \dots, \theta_P\}$ with $\theta_j \sim \text{Uniform}(\Theta)$.
    *   **While** $t < T$:
        *   **For each** agent $\theta_j \in \mathcal{P}$:
            *   $f_j \leftarrow \text{EvaluateCandidate}(\theta_j, \mathcal{D}_{opt}, \mathcal{D}_{val})$
        *   Update population $\mathcal{P}$ using Swarm Intelligence Update Rules (Velocity/Position updates).
        *   $\theta^* \leftarrow \arg\min_{\theta \in \mathcal{P}} f(\theta)$

3.  **Return** $\theta^*$

---

### Function: EvaluateCandidate($\theta, \mathcal{D}_{opt}, \mathcal{D}_{val}$)

1.  **Architecture Initialization**:
    *   Construct Encoder $E_\theta: \mathbb{R}^d \rightarrow \mathbb{R}^z$ and Decoder $D_\theta: \mathbb{R}^z \rightarrow \mathbb{R}^d$.
    *   **Constraint 1 (Unbounded Information Flow)**: The bottleneck layer $h = E_\theta(x)$ must use **Linear Activation** (Identity), i.e., $h \in \mathbb{R}^z$, not $\mathbb{R}^z_{+}$.
    
2.  **Robust Denoising Training**:
    *   Initialize weights $W$. Optimizer: Adam with Weight Decay $\lambda$ (L2 Regularization).
    *   **For** epoch $k = 1$ to $E_{max}$:
        *   Sample batch $\mathcal{B} \subset \mathcal{D}_{opt}$.
        *   **Constraint 2 (Noise Injection)**: $\tilde{x} = x + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2I)$.
        *   Compute Loss: $\mathcal{L} = \frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} \| x - D_\theta(E_\theta(\tilde{x})) \|^2$.
        *   Update weights: $W \leftarrow W - \eta \nabla_W \mathcal{L}$.

3.  **Anomaly Scoring**:
    *   Compute Reconstruction Error vector $\mathbf{r}$ on $\mathcal{D}_{val}$:
        $r_i = \| x_i - D_\theta(E_\theta(x_i)) \|^2, \quad \forall (x_i, y_i) \in \mathcal{D}_{val}$.

4.  **Fitness Calculation (AUPRC)**:
    *   Compute Precision-Recall Curve from scores $\mathbf{r}$ and labels $\mathbf{y}_{val}$.
    *   $Fit_{score} = 1.0 - \text{AUC}_{PR}(\mathbf{r}, \mathbf{y}_{val})$.
    *   **Return** $Fit_{score}$.

---

## Methodological Explanation

This approach introduces three key deviations from standard autoencoder optimization to address the specific challenges of financial fraud detection in highly imbalanced datasets.

### 1. Unbounded Latent Space Representation
Standard autoencoders often use non-linear activations (e.g., ReLU, Sigmoid) at the bottleneck layer. However, when input data is Standard Scaled (containing negative values) and the latent dimension is highly compressed (e.g., $z=4$), constraining the latent space to $\mathbb{R}^+_{z}$ (via ReLU) effectively creates a "dead zone" for half the feature space.
**Novelty**: We enforce a **Linear (Unbounded)** bottleneck. This allows the encoder to utilize the full continuous manifold $\mathbb{R}^z$ to represent legitimate transactions, significantly increasing the information capacity per latent unit and improving the reconstruction of subtle outlier patterns.

### 2. Stratified AUPRC Optimization Objective
Traditional hyperparameter optimization often minimizes Reconstruction Loss (MSE) or maximizes F1-score.
*   *Critique of MSE*: Minimizing MSE focuses on reconstructing the majority class (normal transactions) perfectly, often ignoring the structure of anomalies.
*   *Critique of F1*: F1 is threshold-dependent and unstable during stochastic optimization.
**Novelty**: We utilize **AUPRC (Area Under Precision-Recall Curve)** as the direct optimization objective. By using a **Stratified Validation Set** that preserves the extreme imbalance (e.g., 1:500), the optimizer is forced to select architectures that inherently separate the minority class distribution from the majority, rather than those that simply memorize normal data.

### 3. Denoising Regularization in Evaluation Loop
To prevent the optimizer from selecting extremely large, overfitting networks that memorize the noise in the training sample:
*   **Input Noise Injection**: We corrupt inputs with Gaussian noise ($\sigma=0.1$) during the candidate evaluation phase.
*   **Weight Decay**: We strictly enforce L2 regularization during the candidate training.
**Effect**: This penalizes "memorization" architectures. Only architectures that learn robust, generalized features can reconstruct the clean original input from the corrupted version, leading to a lower reconstruction error on the validation set and a higher AUPRC score.
