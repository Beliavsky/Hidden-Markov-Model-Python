import numpy as np
import pandas as pd

def autocorr(x, lag):
    """ return autocorrelations 1:lag inclusive """
    return np.corrcoef(x[lag:], x[:-lag])[0, 1] if lag < len(x) else np.nan

def print_acf_raw_squared(x, nacf=5):
    """ print autocorrelations of x and x**2 """
    if nacf < 1:
        return
    lags = range(1, nacf + 1)
    acf_raw = [autocorr(x, lag) for lag in lags]
    acf_squared = [autocorr(x**2, lag) for lag in lags]

    # Store autocorrelations in a dataframe with 3 decimal places
    acf_df = pd.DataFrame({
        'Lag': lags,
        'Raw': [round(x, 3) for x in acf_raw],
        'Squared': [round(x, 3) for x in acf_squared]
    })
    print("\nAutocorrelations:")
    print(acf_df.to_string(index=False))