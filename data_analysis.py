import pandas as pd
import os
import re
import matplotlib.pyplot as plt


# Function to check for missing values
def check_missing_values(df):
    return df.isnull().sum().sum()


def check_class_imbalance(df):
    target_col = df.columns[-1]  # Target class is the last column
    class_counts = df[target_col].value_counts()
    return class_counts


# Function to check for invalid characters in column names and data
def check_invalid_characters(df, pattern=r"[^a-zA-Z0-9_ ]"):
    invalid_columns = [col for col in df.columns if re.search(pattern, col)]
    invalid_cells = df.map(lambda x: bool(re.search(pattern, str(x))) if pd.notnull(x) else False).sum().sum()
    return invalid_columns, invalid_cells


# Function to analyze datasets
def analyze_datasets():
    ready_to_use = []
    for i in range(5):
        filename = f"data{i}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            print(f"\nAnalyzing {filename}:")

            missing_values = check_missing_values(df)
            class_imbalance = check_class_imbalance(df)
            invalid_columns, invalid_cells = check_invalid_characters(df)

            print(f"Missing Values: {missing_values}")
            print(f"Invalid Columns: {len(invalid_columns)}")
            print(f"Invalid Cells: {invalid_cells}")
            print(f"Class Distribution:\n{class_imbalance}")


            plt.figure(figsize=(6, 4))
            class_imbalance.plot(kind='bar', color=['blue', 'orange'])
            plt.title(f"Class Distribution for {filename}")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.xticks(rotation=0)
            plt.show()

            if missing_values <= 10 and len(invalid_columns) <= 10 and invalid_cells <= 10:
                print(f"{filename} is ready to use.")
                ready_to_use.append(filename)
            else:
                print(f"{filename} needs cleaning.")

    print("\nFinal Decision:")
    if ready_to_use:
        print(f"Datasets ready to use: {ready_to_use}")
    else:
        print("No dataset is ready to use without cleaning.")


analyze_datasets()
