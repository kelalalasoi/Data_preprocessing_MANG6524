# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 04:47:06 2023

@author: Kelsie Nguyen
"""

import pandas as pd

# Load your dataframe
data = pd.read_excel('E:\Study Abroad\Semester 01\MANG6524 - Marketing Analytics and Visualisations\Final Assessment\Output\Eliminate noise\Df for k_means.xlsx')

# Create a directory to store the separated Excel files
output_directory = 'E:\Study Abroad\Semester 01\MANG6524 - Marketing Analytics and Visualisations\Final Assessment\Output\SeparatedFiles.xlsx'

# Iterate over unique values in the 'Country' column
for country_value in data['Country'].unique():
    # Create a subset for each unique value
    subset = data[data['Country'] == country_value]
    
    # Create an Excel file for each subset
    output_file_path = f'{output_directory}subset_{country_value}.xlsx'
    subset.to_excel(output_file_path, index=False)

print("Separation completed.")
