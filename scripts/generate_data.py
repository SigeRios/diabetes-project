import pandas as pd
import numpy as np

# Function to generate sample data
def generate_sample_data(rows, filename, mean_shift=0):
    np.random.seed(42)
    data = {
        'Feature1': np.random.normal(loc=5 + mean_shift, scale=2, size=rows),
        'Feature2': np.random.normal(loc=10 + mean_shift, scale=5, size=rows),
        'Outcome': np.random.randint(0, 2, size=rows)
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Generate reference data
generate_sample_data(1000, 'data/reference_data.csv')

# Generate new data with a mean shift to simulate drift
generate_sample_data(1000, 'data/new_data.csv', mean_shift=2)
