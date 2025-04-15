"""
Simulate and estimate a two-state Markov-switching process with
autoregressive dynamics. Each state's data is drawn from a p=2
autoregressive model with specified AR coefficients and additive noise
with a specified mean and standard deviation. Fits a Hidden Markov
Model (HMM) using hmmlearn and prints both simulated and estimated
parameters, including the AR(1) and AR(2) coefficients.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from hmmlearn import hmm
from stats import print_acf_raw_squared

# ----------------------------
# Parameters
# ----------------------------
n_steps = 10**6
seed = 42
if seed is not None:
    np.random.seed(seed)
# Noise parameters for each state (mean, std)
mean_state1, std_state1 = 0, 1
mean_state2, std_state2 = 0, 3

# AR(2) coefficients for each state
ar1_state1, ar2_state1 = 0.5, -0.2
ar1_state2, ar2_state2 = 0.3,  0.1

# Transition probabilities for the Markov chain
p11, p22 = 0.9, 0.7
p12, p21 = 1 - p11, 1 - p22

plot_figures = False  # Toggle all plotting

# Dictionary to store timing information
timings = {}

# ----------------------------
# Simulation: Markov Chain and AR(2) Process
# ----------------------------
t0 = time.time()
states = np.zeros(n_steps, dtype=int)
increments = np.zeros(n_steps)
current_state = 1

for i in range(n_steps):
    states[i] = current_state
    if i < 2:
        # For the first two time points no AR terms are available; use simple noise.
        if current_state == 1:
            increments[i] = np.random.normal(mean_state1, std_state1)
        else:
            increments[i] = np.random.normal(mean_state2, std_state2)
    else:
        if current_state == 1:
            increments[i] = (ar1_state1 * increments[i-1] +
                             ar2_state1 * increments[i-2] +
                             np.random.normal(mean_state1, std_state1))
        else:
            increments[i] = (ar1_state2 * increments[i-1] +
                             ar2_state2 * increments[i-2] +
                             np.random.normal(mean_state2, std_state2))
    # Update current state based on transition probabilities.
    if current_state == 1:
        current_state = 1 if np.random.rand() < p11 else 2
    else:
        current_state = 1 if np.random.rand() < p21 else 2

random_walk = np.cumsum(increments)
timings['Simulation'] = time.time() - t0

# ----------------------------
# Simulated Parameters
# ----------------------------
print("#obs:", n_steps)
print("Simulated State Parameters:")
simulated_params_df = pd.DataFrame({
    'State': ['State 1', 'State 2'],
    'Mean': [f"{mean_state1:.4f}", f"{mean_state2:.4f}"],
    'Std Dev': [f"{std_state1:.4f}", f"{std_state2:.4f}"],
    'AR(1)': [f"{ar1_state1:.4f}", f"{ar1_state2:.4f}"],
    'AR(2)': [f"{ar2_state1:.4f}", f"{ar2_state2:.4f}"]
})
print(simulated_params_df)

print("\nSimulated Transition Probabilities:")
simulated_trans_df = pd.DataFrame({
    'From \\ To': ['State 1', 'State 2'],
    'State 1': [f"{p11:.4f}", f"{p21:.4f}"],
    'State 2': [f"{p12:.4f}", f"{p22:.4f}"]
})
print(simulated_trans_df)

# ----------------------------
# Fraction of Time in Each State (Simulated)
# ----------------------------
frac_state1 = np.sum(states == 1) / n_steps
frac_state2 = np.sum(states == 2) / n_steps
frac_df = pd.DataFrame({
    'State': ['State 1', 'State 2'],
    'Fraction of Time': [f"{frac_state1:.4f}", f"{frac_state2:.4f}"]
})
print("\nFraction of Time in Each State (Simulated):")
print(frac_df)

# ----------------------------
# Increments Statistics
# ----------------------------
stats_dict = {
    'Mean': np.mean(increments),
    'Std Dev': np.std(increments),
    'Skew': stats.skew(increments),
    'Kurtosis': stats.kurtosis(increments),
    'Min': np.min(increments),
    'Max': np.max(increments)
}

print("\nIncrements Statistics:")
label_width = max(len(label) for label in stats_dict)
for label, value in stats_dict.items():
    print(f"{label:>{label_width}}: {value:9.4f}")

# ----------------------------
# Fit HMM using hmmlearn
# ----------------------------
t0 = time.time()
X = increments.reshape(-1, 1)
model = hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=1000, random_state=42)
model.fit(X)
timings['HMM Fitting'] = time.time() - t0

# ----------------------------
# Estimated Emission Parameters and AR Coefficients
# ----------------------------
# Sort states by increasing estimated standard deviation from the emission model
t0 = time.time()
sorted_indices = np.argsort([np.sqrt(model.covars_[i][0, 0]) for i in range(2)])
estimated_means = model.means_.flatten()[sorted_indices]
estimated_stds = np.sqrt(np.array([model.covars_[i][0, 0] for i in range(2)]))[sorted_indices]

# Estimate AR coefficients using OLS on segments where the predicted state is constant.
hidden_states = model.predict(X)
ar_coeffs_est = {}
for s in sorted_indices:
    X_list = []
    y_list = []
    for i in range(2, n_steps):
        if hidden_states[i] == s and hidden_states[i-1] == s and hidden_states[i-2] == s:
            X_list.append([increments[i-1], increments[i-2]])
            y_list.append(increments[i])
    if len(y_list) > 0:
        X_arr = np.array(X_list)
        y_arr = np.array(y_list)
        # Estimate coefficients: x[t] = phi1*x[t-1] + phi2*x[t-2]
        phi, _, _, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        ar_coeffs_est[s] = phi
    else:
        ar_coeffs_est[s] = np.array([np.nan, np.nan])
timings['AR Estimation'] = time.time() - t0

estimated_ar1 = [f"{ar_coeffs_est[s][0]:.4f}" for s in sorted_indices]
estimated_ar2 = [f"{ar_coeffs_est[s][1]:.4f}" for s in sorted_indices]

estimated_params_df = pd.DataFrame({
    'State': [f"State {i+1}" for i in range(2)],
    'Mean': [f"{estimated_means[i]:.4f}" for i in range(2)],
    'Std Dev': [f"{estimated_stds[i]:.4f}" for i in range(2)],
    'AR(1)': estimated_ar1,
    'AR(2)': estimated_ar2
})
print("\nEstimated State Parameters:")
print(estimated_params_df)

estimated_transmat = model.transmat_[np.ix_(sorted_indices, sorted_indices)]
estimated_trans_df = pd.DataFrame({
    'From \\ To': [f"State {i+1}" for i in range(2)],
    'State 1': [f"{row[0]:.4f}" for row in estimated_transmat],
    'State 2': [f"{row[1]:.4f}" for row in estimated_transmat]
})
print("\nEstimated Transition Probabilities:")
print(estimated_trans_df)

print_acf_raw_squared(increments, nacf=0)

# ----------------------------
# Print Timing Summary
# ----------------------------
print("\nTiming Summary (excluding plotting):")
for key, value in timings.items():
    print(f"{key:>15}: {value:.4f} seconds")

# ----------------------------
# Plotting
# ----------------------------
if plot_figures:
    hidden_states = model.predict(X)
    plt.figure(figsize=(12, 5))
    plt.plot(hidden_states, drawstyle="steps-post", label="Inferred Hidden States", alpha=0.8)
    plt.plot(states, drawstyle="steps-post", label="True States", alpha=0.6)
    plt.xlabel("Time Step")
    plt.ylabel("State")
    plt.title("Comparison of Inferred Hidden States and True States")
    plt.legend()
    plt.grid(True)
    plt.show()
