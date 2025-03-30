import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Manually specify the files for each category
datasets = {
    "0.7_early": [
        "average_data/0.7pct_Agar_(20x)_dataset_1_bin_data_0.7_1.7_early.csv",
        "average_data/0.7pct_Agar_(20x)_dataset_2_bin_data_0.7_1.7_early.csv",
        "average_data/0.7pct_Agar_(20x)_dataset_3_bin_data_0.7_1.7_early.csv",
    ],
    "0.7_late": [
        "average_data/0.7pct_Agar_(10x)_dataset_1_bin_data_0.7_1.7_late.csv",
        "average_data/0.7pct_Agar_(10x)_dataset_2_bin_data_0.7_1.7_late.csv",
        "average_data/0.7pct_Agar_(10x)_dataset_3_bin_data_0.7_1.7_late.csv",
    ],
    "1.7_early": [
        "average_data/1.7pct_Agar_(30x)_dataset_1_bin_data_0.7_1.7_early.csv",
        "average_data/1.7pct_Agar_(30x)_dataset_2_bin_data_0.7_1.7_early.csv",
        "average_data/1.7pct_Agar_(30x)_dataset_3_bin_data_0.7_1.7_early.csv",
    ],
    "1.7_late": [
        "average_data/1.7pct_Agar_(20x)_dataset_1_bin_data_0.7_1.7_late.csv",
        "average_data/1.7pct_Agar_(20x)_dataset_2_bin_data_0.7_1.7_late.csv",
        "average_data/1.7pct_Agar_(20x)_dataset_3_bin_data_0.7_1.7_late.csv",
    ],
}

# Define colors for the datasets
colors = {
    "0.7": ["#1f77b4", "#4a90e2", "#87ceeb"],  # Shades of blue
    "1.7": ["#d62728", "#ff6347", "#ff9999"],  # Shades of red
}

# Function to load data from a CSV file
def load_data(file_name):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_name)

    # Extract the bin centers and average dot products
    bin_centers = data["Bin Center (µm)"].values  # Convert to NumPy array
    avg_dot_products = data["Average Dot Product"].values  # Convert to NumPy array

    return bin_centers, avg_dot_products

# Function to calculate global y-axis limits
def calculate_global_y_limits(datasets):
    global_y_min = float("inf")
    global_y_max = float("-inf")

    for category, files in datasets.items():
        for file_name in files:
            # Load data from the CSV file
            _, avg_dot_products = load_data(file_name)

            # Update global min and max
            global_y_min = min(global_y_min, avg_dot_products.min())
            global_y_max = max(global_y_max, avg_dot_products.max())

    return global_y_min, global_y_max

# Calculate global y-axis limits
global_y_min, global_y_max = calculate_global_y_limits(datasets)

# Function to plot a single graph
def plot_graph(category, files, color_shades, title):
    plt.figure(figsize=(8, 6))
    max_bin_center = 0  # Initialize the maximum bin center

    for i, file_name in enumerate(files, start=1):
        # Load data from the CSV file
        bin_centers, avg_dot_products = load_data(file_name)

        # Update the maximum bin center
        max_bin_center = max(max_bin_center, bin_centers.max())

        # Plot the data
        plt.plot(bin_centers, avg_dot_products, label=f"Dataset {i}", color=color_shades[i - 1], linewidth=1.5)

    # Set plot labels and title
    plt.xlim(0, max_bin_center / 2)  # Set x-axis limit to half the maximum bin center
    plt.ylim(global_y_min, global_y_max)  # Use global y-axis limits
    plt.xlabel(r"Distance ($\mu m$)")
    plt.ylabel("Average Dot Product Squared")
    plt.title(title)
    plt.legend(title="Dataset")  # Add a title to the legend
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# Plot each category in a separate window
plot_graph("0.7_early", datasets["0.7_early"], colors["0.7"], "0.7% Agar Early Data")
plot_graph("0.7_late", datasets["0.7_late"], colors["0.7"], "0.7% Agar Late Data")
plot_graph("1.7_early", datasets["1.7_early"], colors["1.7"], "1.7% Agar Early Data")
plot_graph("1.7_late", datasets["1.7_late"], colors["1.7"], "1.7% Agar Late Data")