# Solution code for the Iris Dataset Homework (run.py)

import pandas as pd
from scipy.stats import zscore

# Question 1: Pre-process the data
def preprocess_data(input_filename):
    # Load the dataset and assign column labels
    data = pd.read_csv(input_filename, names=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"], header=0)
    
    # Calculate z-scores for SepalLengthCm and SepalWidthCm
    data['SepalLengthCm_z'] = zscore(data['SepalLengthCm'])
    data['SepalWidthCm_z'] = zscore(data['SepalWidthCm'])
    
    # Filter out outliers with z-scores less than -2 or greater than 2
    filtered_data = data[(data['SepalLengthCm_z'].abs() <= 2) & (data['SepalWidthCm_z'].abs() <= 2)].copy()
    
    # Add an ID column as a unique identifier for each row
    filtered_data.loc[:, 'ID'] = range(1, len(filtered_data)+1)
    
    # Drop z-score columns for final output
    return filtered_data.drop(columns=['SepalLengthCm_z', 'SepalWidthCm_z'])



# Question 2: Descriptive Statistics Functions
def species_count(data):
    preprocessed_data = preprocess_data(data)
    return preprocessed_data['Species'].value_counts().to_dict()

def average_sepal_length(data):
    preprocessed_data = preprocess_data(data)
    return round((preprocessed_data['SepalLengthCm'].mean()),1)

def max_petal_width(data):
    preprocessed_data = preprocess_data(data)
    return preprocessed_data['PetalWidthCm'].max()

def min_petal_length(data):
    preprocessed_data = preprocess_data(data)
    return preprocessed_data['PetalLengthCm'].min()

def count_sepal_length_above_5(data):
    preprocessed_data = preprocess_data(data)
    return len(preprocessed_data[preprocessed_data['SepalLengthCm'] > 5.0])

# Question 3: Analysis Functions
def count_petal_length_below_2(data):
    preprocessed_data = preprocess_data(data)
    return len(preprocessed_data[preprocessed_data['PetalLengthCm'] < 2.0])

def get_sepal_width_above_3_5(data):
    preprocessed_data = preprocess_data(data)
    return sorted(preprocessed_data[preprocessed_data['SepalWidthCm'] > 3.5]['ID'])

def species_count_petal_width_above_1_5(data):
    preprocessed_data = preprocess_data(data)
    filtered_data = preprocessed_data[preprocessed_data['PetalWidthCm'] > 1.5]
    return filtered_data['Species'].value_counts().to_dict()

def get_virginica_petal_length_above_6(data):
    preprocessed_data = preprocess_data(data)
    virginica_data = preprocessed_data[(preprocessed_data['Species'] == 'Iris-virginica') & (preprocessed_data['PetalLengthCm'] > 6.0)]
    return sorted(virginica_data['ID'])

def get_largest_sepal_width(data):
    preprocessed_data = preprocess_data(data)
    return preprocessed_data.loc[preprocessed_data['SepalWidthCm'].idxmax()]["ID"]

