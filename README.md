# FPT-Project-Data-Profilling-Tool
DataProfiler is a Python library designed to generate comprehensive statistical reports and visualizations for CSV data files. It provides insights into the structure and characteristics of your dataset, helping you understand and analyze your data quickly and effectively.

## Installation
To install DataProfiler, simply download the dataprofiler.py file and place it in your desired directory.

## Usage
DataProfiler requires Python 3 and the following dependencies:

- pandas
- matplotlib
- seaborn
- scipy
- numpy

You can install these dependencies using pip:
`pip install numpy pandas matplotlib scipy seaborn`

Once installed, you can use DataProfiler in your Python code as follows:
```
import pandas as pd
import dataprofiler

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('train.csv')

# Generate a profile report
dataprofiler.profileReport(df, title='Train data')
```

This will produce an HTML report titled "Train data" containing insights and visualizations for your train.csv dataset.
