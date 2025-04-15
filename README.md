# Hidden-Markov-Model-Python
Simulate and fit a Markov-switching process with autoregressive dynamics, using hmmlearn for estimation.<br> 
Output of `python xsim_fit_hmm_ar.py`:

```
#obs: 1000000
Simulated State Parameters:
     State    Mean Std Dev   AR(1)    AR(2)
0  State 1  0.0000  1.0000  0.5000  -0.2000
1  State 2  0.0000  3.0000  0.3000   0.1000

Simulated Transition Probabilities:
  From \ To State 1 State 2
0   State 1  0.9000  0.1000
1   State 2  0.3000  0.7000

Fraction of Time in Each State (Simulated):
     State Fraction of Time
0  State 1           0.7504
1  State 2           0.2496

Increments Statistics:
    Mean:   -0.0034
 Std Dev:    1.8896
    Skew:   -0.0124
Kurtosis:    3.1537
     Min:  -14.0707
     Max:   14.2462

Estimated State Parameters:
     State     Mean Std Dev   AR(1)    AR(2)
0  State 1   0.0004  1.1121  0.3910  -0.1413
1  State 2  -0.0130  3.0691  0.4273   0.0109

Estimated Transition Probabilities:
  From \ To State 1 State 2
0   State 1  0.9105  0.0895
1   State 2  0.2243  0.7757

Timing Summary (excluding plotting):
     Simulation: 1.8226 seconds
    HMM Fitting: 24.7409 seconds
  AR Estimation: 2.5944 seconds
```
