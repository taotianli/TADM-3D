import sys
sys.path.append('.')
import pandas as pd
from utilities.dataset import SRDataSet
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help="Path to the dataset directory")
    parser.add_argument('--output_dir', required=True, type=str, help="Directory to save the output CSV file")
    parser.add_argument('--prefix', required=False, type=str, default='train', help="Prefix for the dataset (default: train)")
    args = parser.parse_args()

    # Initialize the dataset
    data_path = args.dataset
    prefix = args.prefix
    dataset = SRDataSet(data=data_path, prefix=prefix)

    # List to hold data rows
    data_rows = []

    # Total length of the dataset for progress tracking
    total_items = len(dataset)

    # Iterate through the dataset to extract relevant fields
    for index in range(total_items):
        item = dataset[index]
        if item['item_name'] != "empty":  # Exclude empty items
            data_rows.append({
                'img_hr': item['img_hr'],
                'img_lr': item['img_lr'],
                'item_name': item['item_name'],
                'diff_ages': item['diff_ages'],
                'patient_condition': item['patient_condition'],
                'age': item['age'],
                'split': item['split']
            })

        # Print progress every 100 items
        if (index + 1) % 100 == 0 or (index + 1) == total_items:
            print(f"Processed {index + 1}/{total_items} items...")

    # Create a DataFrame from the data rows
    df = pd.DataFrame(data_rows)

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the DataFrame to a CSV file
    output_csv_path = os.path.join(args.output_dir, f"{prefix}_dataset.csv")
    df.to_csv(output_csv_path, index=False)

    print(f"CSV file created at {output_csv_path}")
