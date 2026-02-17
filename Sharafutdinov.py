import pandas as pd
import numpy as np

file_old = 'music_old.csv'
file_new = 'music_new.csv'

df_old = pd.read_csv(file_old)
df_new = pd.read_csv(file_new)

old_loudness = df_old['loudness'].dropna()
new_loudness = df_new['loudness'].dropna()

def calculate_psi(old_data, new_data, number_of_bins=20):
    bin_edges = []
    for i in range(number_of_bins + 1):
        percentile_value = (100 / number_of_bins) * i
        bin_edges.append(np.percentile(old_data, percentile_value))
    
    bin_edges[-1] = bin_edges[-1] + 0.0001
    
    old_counts = [0] * number_of_bins
    for value in old_data:
        for i in range(number_of_bins):
            if bin_edges[i] <= value < bin_edges[i + 1]:
                old_counts[i] = old_counts[i] + 1
                break
    
    new_counts = [0] * number_of_bins
    for value in new_data:
        for i in range(number_of_bins):
            if bin_edges[i] <= value < bin_edges[i + 1]:
                new_counts[i] = new_counts[i] + 1
                break
    
    old_percents = []
    for count in old_counts:
        old_percents.append(count / len(old_data))
    
    new_percents = []
    for count in new_counts:
        new_percents.append(count / len(new_data))
    
    for i in range(number_of_bins):
        if old_percents[i] == 0:
            old_percents[i] = 0.0001
        if new_percents[i] == 0:
            new_percents[i] = 0.0001
    
    psi_value = 0
    for i in range(number_of_bins):
        psi_value = psi_value + (new_percents[i] - old_percents[i]) * np.log(new_percents[i] / old_percents[i])
    
    return psi_value

psi_result = calculate_psi(old_loudness, new_loudness, 20)
print(f"{psi_result:.4f}")