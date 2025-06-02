import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse.linalg import eigsh
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import math

seed = 14
np.random.seed(seed)
torch.manual_seed(seed)

def training(denoiser, 
             samples, 
             num_epochs=200, 
             criterion = nn.MSELoss(), 
             lr=1e-3, 
             weight_decay=1e-4,
             alpha_min = 0,
             alpha_max = np.pi/2):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=lr, weight_decay=weight_decay)

    batch_size = 50
    device = torch.device("cpu")

    best_val_loss = float('inf')
    best_model_state = None
    true_loss = 0

    samples_train, samples_val = train_test_split(samples, test_size=0.4, random_state=42)
    train_tensor = torch.tensor(samples_train, dtype=torch.float32)
    val_tensor = torch.tensor(samples_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor, val_tensor), batch_size=5, shuffle=False)

    for epoch in range(num_epochs):
        denoiser.train()
        running_loss = 0.0

        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            optimizer.zero_grad()

            alpha_vals = np.random.uniform(alpha_min, alpha_max, size=(batch_X.shape[0],))
            alpha = torch.tensor(alpha_vals, dtype=torch.float32, device=device).unsqueeze(1)
            t = torch.tan(alpha).pow(2)

            noise = torch.randn_like(batch_X)
            input = (t * batch_X + t.sqrt() * noise)
            target = batch_X

            output = denoiser(input, alpha.squeeze(1))
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # === VALIDATION ===
        denoiser.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)

                alpha_vals = np.random.uniform(alpha_min, alpha_max, size=(batch_X.shape[0],))
                alpha = torch.tensor(alpha_vals, dtype=torch.float32, device=device).unsqueeze(1)
                t = torch.tan(alpha).pow(2)

                noise = torch.randn_like(batch_X)
                input = (t * batch_X + t.sqrt() * noise)
                target = batch_X

                output = denoiser(input, alpha.squeeze(1))

                # Empirical relative error
                rel_mse = (((output - target) ** 2).sum(dim=1) / (target ** 2).sum(dim=1).clamp(min=1e-8)).sqrt()
                val_loss += rel_mse.mean().item()


        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = denoiser.state_dict()


        if epoch % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss*100:.4f}% | ")
            
    denoiser.load_state_dict(best_model_state)
    print(f"Loaded best model with validation loss: {best_val_loss*100:.4f}%")
    return 0


def construct_dataset(n, p, a, num_samples):
    mu1 = (1 - p) * a
    mu2 = -p * a
    cov = np.eye(n)

    # Sample from the mixture model
    components = np.random.choice([0, 1], size=num_samples, p=[p, 1 - p])
    samples = np.array([
        multivariate_normal.rvs(mean=mu1 if c == 0 else mu2, cov=cov)
        for c in components
    ])
    return samples

def forward_sampling(alpha_max, K, denoiser, n, num_samples_target=5000, path=True, frequency=20):
    alpha_seq = torch.linspace(0, alpha_max, K, dtype=torch.float32)

    T = torch.tan(alpha_seq[-1]) ** 2
    print(f"T max: {T.item():.2f}")

    # Initialize target samples (starts at zero)
    target_samples = torch.zeros((num_samples_target, n), dtype=torch.float32)

    # For storing intermediate results
    trajectory_samples = []
    trajectory_alphas = []

    for it in range(K - 1):
        # Current and next time (in transformed space)
        t_curr = torch.tan(alpha_seq[it]) ** 2
        t_next = torch.tan(alpha_seq[it + 1]) ** 2
        delta_t = t_next - t_curr

        alpha = alpha_seq[it]
        alpha_batch = torch.full((num_samples_target,), alpha)

        # Optionally save intermediate state
        if (it + 1) % frequency == 0:
            trajectory_samples.append(target_samples.clone())
            trajectory_alphas.append(alpha)
            if path:
                print(f"Saving y at time: {t_curr.item():.2f}")

        noise = torch.randn((num_samples_target, n), dtype=torch.float32)

        drift = denoiser(target_samples, alpha_batch)

        target_samples = target_samples + delta_t * drift + torch.sqrt(torch.tensor(delta_t)) * noise

    # Final rescaling
    target_samples = target_samples / T

    if path:
        return [target_samples, trajectory_samples, trajectory_alphas]
    return target_samples

def forward_sampling_mixture_aware(alpha_max, K, denoiser_plus, denoiser_minus, n, q, num_samples_target=5000, path=True, frequency=20):
    alpha_it = torch.linspace(0, alpha_max, K, dtype=torch.float32)
    T = torch.tan(alpha_it[-1])**2
    print("Final time T:", T.item())

    # --- First component: mode +a ---
    num_samples_plus = int(np.floor(num_samples_target * q))
    target_samples_plus = torch.zeros((num_samples_plus, n), dtype=torch.float32)

    for it in range(K - 1):
        alpha = alpha_it[it]
        t = torch.tan(alpha)**2
        delta_t = torch.tan(alpha_it[it + 1])**2 - t
        alpha_vec = torch.full((num_samples_plus,), alpha)

        noise = torch.randn((num_samples_plus, n), dtype=torch.float32)
        drift = denoiser_plus(target_samples_plus, alpha_vec)

        target_samples_plus += delta_t * drift + torch.sqrt(torch.tensor(delta_t)) * noise

    target_samples_plus /= T

    # --- Second component: mode -a ---
    num_samples_minus = int(np.floor(num_samples_target * (1 - q)))
    target_samples_minus = torch.zeros((num_samples_minus, n), dtype=torch.float32)

    for it in range(K - 1):
        alpha = alpha_it[it]
        t = torch.tan(alpha)**2
        delta_t = torch.tan(alpha_it[it + 1])**2 - t
        alpha_vec = torch.full((num_samples_minus,), alpha)

        noise = torch.randn((num_samples_minus, n), dtype=torch.float32)
        drift = denoiser_minus(target_samples_minus, alpha_vec)

        target_samples_minus += delta_t * drift + torch.sqrt(torch.tensor(delta_t)) * noise

    target_samples_minus /= T

    # --- Combine both components ---
    target_samples = torch.cat((target_samples_minus, target_samples_plus))
    return target_samples

def forward_sampling_second_phase(target_samples, alpha_min, alpha_max, K, denoiser, n, num_samples_target=5000):
    alpha_seq = torch.linspace(alpha_min, alpha_max, K, dtype=torch.float32)

    T = torch.tan(alpha_seq[-1]) ** 2
    print(f"T max: {T.item():.2f}")



    for it in range(K - 1):
        # Current and next time (in transformed space)
        t_curr = torch.tan(alpha_seq[it]) ** 2
        t_next = torch.tan(alpha_seq[it + 1]) ** 2
        delta_t = t_next - t_curr

        alpha = alpha_seq[it]
        alpha_batch = torch.full((num_samples_target,), alpha)

        noise = torch.randn((num_samples_target, n), dtype=torch.float32)

        drift = denoiser(target_samples, alpha_batch)

        target_samples = target_samples + delta_t * drift + torch.sqrt(torch.tensor(delta_t)) * noise

    # Final rescaling
    target_samples = target_samples / T
    return target_samples
                

def plot_projected_dataset(samples, a, p, n, ax=None):
    """
    Projects the samples onto vector a and plots histogram + theoretical PDF on the provided axis.
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()

    # Compute projection ⟨X_t, a⟩ / ||a||²
    a_norm_sq = np.linalg.norm(a) ** 2
    projections = np.dot(samples, a) / a_norm_sq

    if ax is None:
        ax = plt.gca()

    # Histogram
    ax.hist(projections, bins=100, density=True, alpha=0.5, color='blue', label="Empirical Distribution")

    # Theoretical mixture of Gaussians
    x = np.linspace(min(projections), max(projections), 1000)
    pdf = p * multivariate_normal.pdf(x, mean=(1 - p), cov=1 / n) + \
          (1 - p) * multivariate_normal.pdf(x, mean=-p, cov=1 / n)

    ax.plot(x, pdf, color='red', label="Theoretical PDF")
    ax.set_xlabel(r'$\langle X_t, a \rangle / \|a\|^2$')
    ax.set_ylabel('Density')
    ax.set_title('Projection of Samples')
    ax.legend()
    ax.grid()

def compute_posterior_mean_multidim_batch(y_t, t, mu1, mu2, w):
    """
    Compute E[X | y_t] for a batch of y_t vectors in R^n.
    
    Parameters:
    - y_t: np.array shape (batch_size, n)
    - mu1, mu2: np.array shape (n,)
    - w: scalar in (0,1)
    - t: scalar

    Returns:
    - posterior_mean: np.array shape (batch_size, n)
    """
    cov = t * (t + 1)

    # Ensure mu1 and mu2 are broadcastable
    mu1 = mu1.reshape(1, -1)  # shape: (1, n)
    mu2 = mu2.reshape(1, -1)

    delta1 = y_t - t * mu1  # shape: (batch_size, n)
    delta2 = y_t - t * mu2

    logp1 = -0.5 * np.sum(delta1**2, axis=1) / cov  # shape: (batch_size,)
    logp2 = -0.5 * np.sum(delta2**2, axis=1) / cov

    max_logp = np.maximum(logp1, logp2)
    p1 = w * np.exp(logp1 - max_logp)
    p2 = (1 - w) * np.exp(logp2 - max_logp)
    Z = p1 + p2

    gamma1 = (p1 / Z).reshape(-1, 1)  # shape: (batch_size, 1)
    gamma2 = (p2 / Z).reshape(-1, 1)

    # Posterior mean
    posterior_mean = (1 / (t + 1)) * y_t + (1 / (t + 1)) * (gamma1 * mu1 + gamma2 * mu2)
    return posterior_mean


def evaluate_denoiser_on_gaussian_mixture(
    denoiser, samples, n = 128, p=0.7,
    K=200, plot_every=20, alpha_min = 0.1, alpha_max = np.pi / 2 - 0.1,
):
    print(1)
    a = np.ones(n)
    mu = a
    mu1 = (1 - p) * a
    mu2 = -p * a
    a_norm_sq = np.linalg.norm(mu) ** 2
    alpha_grid = torch.linspace(alpha_min, alpha_max, K, dtype=torch.float32)
    plot_indices = list(range(0, K, plot_every))
    
    cols = 4
    rows = math.ceil(len(plot_indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = axes.flatten()

    mse_list = []
    t_list = []

    samples = torch.tensor(samples, dtype=torch.float32)
    num_samples = samples.shape[0]

    for i in range(K):
        alpha = alpha_grid[i]
        t = torch.tan(alpha).pow(2)
        t_np = t.detach().numpy()
        alpha_batch = torch.full((num_samples,), alpha)

        noise = torch.randn_like(samples)
        y_t = t * samples + t.sqrt() * noise
        y_np = y_t.detach().numpy()

        true_output = compute_posterior_mean_multidim_batch(y_np, t_np, mu1, mu2, p)
        nn_output = denoiser(y_t, alpha_batch).detach().numpy()

        rel_mse = np.sqrt(
            np.sum((nn_output - true_output) ** 2, axis=1) /
            np.clip(np.sum(true_output ** 2, axis=1), a_min=1e-1, a_max=None)
        )
        mse_list.append(rel_mse.mean().item())
        t_list.append(t_np)

        if i in plot_indices:
            true_proj = np.dot(true_output, mu) / a_norm_sq
            nn_proj = np.dot(nn_output, mu) / a_norm_sq
            y_proj = (np.dot(y_np, mu) / a_norm_sq)/t

            ax = axes[plot_indices.index(i)]
            ax.plot(y_proj, true_proj, '.', color='blue', label='True')
            ax.plot(y_proj, nn_proj, '.', color='red', label='NN', alpha=0.1)
            ax.set_xlabel(r'Projection of $y_t/t$ onto $\mu$')
            ax.set_ylabel('Projected output')
            ax.set_title(f'Plot {i+1} — $t$ = {t:.2f}')
            ax.text(
                0.5, -0.25, f'MSE: {rel_mse.mean() * 100:.2f}%',
                transform=ax.transAxes, fontsize=15,
                ha='center', va='top'
            )
            ax.grid(True)
            ax.legend()

    # Hide unused subplots
    for j in range(len(plot_indices), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    # Plot MSE vs t
    plt.figure(figsize=(8, 5))
    plt.plot(t_list, mse_list, marker='o')
    plt.xlabel(r'$t$')
    plt.ylabel('Relative MSE (%)')
    plt.title('Relative MSE vs. Time $t$')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return t_list, mse_list

def splidataset_on_cov(samples):
    C = np.cov(samples, rowvar=False)
    # Compute the principal eigenvector using Lanczos (ARPACK)
    eigvals, eigvecs = eigsh(C, k=1, which='LM')
    principal_eigenvector = eigvecs[:, 0]

    # Fix sign ambiguity by enforcing a consistent sign
    if principal_eigenvector[np.argmax(np.abs(principal_eigenvector))] < 0:
        principal_eigenvector *= -1  # Flip the sign

    # Compute the condition number (ratio of largest to smallest eigenvalue)
    eigvals_full = np.linalg.eigvalsh(C)
    #condition_number = eigvals_full[-1] / eigvals_full[0]
    #print(condition_number)

    samples_plus = samples[samples@principal_eigenvector>=0]
    samples_minus = samples[samples@principal_eigenvector<0]
    return [samples_plus,samples_minus]