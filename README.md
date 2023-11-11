# Airplane Satisfaction Prediction Project

## Overview

This project focuses on predicting passenger satisfaction with airline services using a decision tree classifier implemented from scratch in Python. The decision tree is built based on entropy and information gain criteria.

## Files and Structure

- **`DT2.py`**: Contains the implementation of the Decision Tree algorithm.
- **`airplane_data.csv`**: The dataset used for training and testing the model.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Matplotlib (for visualization)
  
## Usage

1. **Install Dependencies:**
   ```bash
   pip install numpy pandas matplotlib
2. **Run the Project:**
   ```bash
   python DT2.py
3. **Results:**
   - The decision tree structure and relevant information will be printed.
   - Accuracy and other metrics will be displayed based on the testing dataset.

## Customization
  - Adjust hyperparameters and settings in the DecisionTree class in DT2.py.
  - Explore different metrics and features for potential improvements.

## Data
  - The dataset (airplane_data.csv) contains columns such as Gender, Customer Type, Age, Flight Distance, and others.
  - The target variable is 'satisfaction' indicating passenger satisfaction.

## Acknowledgments
  - This project was inspired by the need to predict and understand factors affecting passenger satisfaction in the aviation industry.
