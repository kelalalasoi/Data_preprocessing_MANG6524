# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:41:39 2023

@author: kelsie nguyen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the CSV file into a DataFrame
file_path = 'E:\Study Abroad\Semester 01\MANG6524 - Marketing Analytics and Visualisations\Final Assessment\Input\MANG6524 Coursework Dataset.xlsx'
df = pd.read_excel(file_path)

# Number of Observations and Variables
num_observations = df.shape[0]
num_variables = df.shape[1]
print(f"Number of Observations: {num_observations}")
print(f"Number of Variables: {num_variables}")

# Data Types
data_types = df.dtypes

print("Data Types:")
for col, dtype in data_types.items():
    print(f"{col}: {dtype}")
    
#Check for duplicates
duplicates = df.duplicated()
print(df[duplicates])

# Identify Missing Values
missing_values = df.isnull().sum()

print("\nMissing Values:")
for col, missing in missing_values.items():
    print(f"{col}: {missing}")

# Decide on Handling Strategy
# In this case, let's drop rows with missing values
df = df.dropna()

# After handling missing values, we can check again the number of observations
num_observations_after_handling = df.shape[0]

print(f"\nNumber of Observations After Handling Missing Values: {num_observations_after_handling}")

#Check summary of int64 data type
def clean_df(df):
    cleaned_df = pd.DataFrame()

    for column in df.columns:
        if df[column].dtype == 'int64':
            cleaned_df[column] = df[column]

    return cleaned_df
df = clean_df(df)

# Calculate Mean
df_mean = df.mean()

# Calculate Median
df_median = df.median()

# Calculate Standard Deviation
df_std = df.std()

# Calculate Min
df_min = df.min()

# Calculate Max
df_max = df.max()

print("Mean:")
print(df_mean)

print("\nMedian:")
print(df_median)

print("\nStandard Deviation:")
print(df_std)

print("\nMin:")
print(df_min)

print("\nMax:")
print(df_max)

def normality_check(df):
    # Creating a dictionary to store p-values
    p_values = {}

    # Checking normality for each numeric variable
    for column in df.select_dtypes(include='number').columns:
        stat, p = stats.normaltest(df[column].dropna())
        p_values[column] = p

    return p_values

def plot_normality(df, p_values, threshold=0.05):
    for column in df.select_dtypes(include='number').columns:
        if p_values[column] < threshold:
            plt.figure(figsize=(12, 4))

            # Histogram
            plt.subplot(1, 2, 1)
            df[column].hist(bins=30)
            plt.title(f'Histogram of {column}')

            # Q-Q Plot
            plt.subplot(1, 2, 2)
            stats.probplot(df[column].dropna(), dist='norm', plot=plt)
            plt.title(f'Q-Q Plot of {column}')

            plt.show()

def normality_check(df):
    p_values = {}
    for column in df.select_dtypes(include='number').columns:
        stat, p_value = stats.normaltest(df[column].dropna())
        p_values[column] = p_value
    return p_values

# Performing normality check
p_values = normality_check(df)

# Plotting normality checks
plot_normality(df, p_values)

# Identify extreme values
z_scores = np.abs(stats.zscore(df))
extreme_indices = np.where(z_scores > 3)

# Create a new DataFrame with the extreme values
extreme_values = df.iloc[extreme_indices]

# Create a new DataFrame with the clean data
cleaned_df = df[(z_scores <= 3).all(axis=1)]

with pd.ExcelWriter('E:\Study Abroad\Semester 01\MANG6524 - Marketing Analytics and Visualisations\Final Assessment\Output\Eliminate noise\Df after z-score.xlsx') as writer:
    cleaned_df.to_excel(writer, sheet_name='Clean Data', index=False)

# Load your dataset for data analysis
file_path_analysis = 'E:\Study Abroad\Semester 01\MANG6524 - Marketing Analytics and Visualisations\Final Assessment\Output\Eliminate noise\Df after z-score.xlsx'
df_analysis = pd.read_excel(file_path_analysis)

#Summarise the dataset group by country

analysis_num_observations = df_analysis.shape[0]
analysis_num_variables = df_analysis.shape[1]
print(f"Number of Observations: {analysis_num_observations}")
print(f"Number of Variables: {analysis_num_variables}")




