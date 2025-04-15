"""
Simulate and estimate a two-state Markov-switching process with Gaussian emissions.
Each state emits normal increments with different standard deviations.
Fits a Hidden Markov Model (HMM) using hmmlearn and prints estimated parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from hmmlearn import hmm
from stats import print_acf_raw_squared

# ----------------------------
# Parameters
# ----------------------------
n_steps = 10000
mean_state1, std_state1 = 0, 1
mean_state2, std_state2 = 0, 3
p11, p22 = 0.99, 0.7
p12, p21 = 1 - p11, 1 - p22

plot_figures = False  # Toggle all plotting

# ----------------------------
# Simulate Markov Chain and Random Walk
# ----------------------------
states = np.zeros(n_steps, dtype=int)
increments = np.zeros(n_steps)
current_state = 1

for i in range(n_steps):
    states[i] = current_state
    if current_state == 1:
        increments[i] = np.random.normal(mean_state1, std_state1)
    else:
        increments[i] = np.random.normal(mean_state2, std_state2)

    if current_state == 1:
        current_state = 1 if np.random.rand() < p11 else 2
    else:
        current_state = 1 if np.random.rand() < p21 else 2

random_walk = np.cumsum(increments)

# ----------------------------
# Simulated Parameters
# ----------------------------
print("#obs:", n_steps)
print("Simulated State Parameters:")
simulated_params_df = pd.DataFrame({
    'State': ['State 1', 'State 2'],
    'Mean': [f"{mean_state1:.4f}", f"{mean_state2:.4f}"],
    'Std Dev': [f"{std_state1:.4f}", f"{std_state2:.4f}"]
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
X = increments.reshape(-1, 1)
model = hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=1000, random_state=42)
model.fit(X)

# ----------------------------
# Estimated Parameters in Matching Format
# ----------------------------
# Sort states by increasing standard deviation for consistent labeling
sorted_indices = np.argsort([np.sqrt(model.covars_[i][0, 0]) for i in range(2)])
estimated_means = model.means_.flatten()[sorted_indices]
estimated_stds = np.sqrt(np.array([model.covars_[i][0, 0] for i in sorted_indices]))

estimated_params_df = pd.DataFrame({
    'State': [f"State {i+1}" for i in range(2)],
    'Mean': [f"{estimated_means[i]:.4f}" for i in range(2)],
    'Std Dev': [f"{estimated_stds[i]:.4f}" for i in range(2)]
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
