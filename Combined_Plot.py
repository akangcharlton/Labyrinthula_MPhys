import numpy as np  # For numerical operations
import pandas as pd  # For handling CSV data
import matplotlib.pyplot as plt  # For plotting graphs
import os  # For file and directory operations

# Function to compute distances and dot products (including self-pairs)
def compute_dot_product_and_distance(positions, displacements):
    """
    Compute pairwise distances and normalized dot products, including self-comparison.

    Parameters:
        positions (ndarray): Array of (X, Y) positions.
        displacements (ndarray): Array of (DX, DY) displacement vectors.

    Returns:
        distances (ndarray): Array of pairwise distances.
        dot_products (ndarray): Array of normalized dot products squared.
        dot_prods (ndarray): Array of raw dot products.
    """
    n = len(positions)  # Number of points
    X = positions[:, 0]  # X-coordinates
    Y = positions[:, 1]  # Y-coordinates
    distances = []  # List to store distances
    dot_products = []  # List to store normalized dot products squared
    dot_prods = []  # List to store raw dot products

    # Iterate over all pairs of points
    for i in range(n):
        for j in range(n):  # Include self-distances
            # Compute Euclidean distance between points i and j
            dist = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2)
            
            if i == j:
                # A vector's dot product with itself is always 1
                dot_prod = 1
                dot_product2 = 1
            else:
                # Compute the dot product of displacement vectors
                dot_prod = np.dot(np.abs(displacements[i]), np.abs(displacements[j]))
                # Compute the normalized dot product squared
                dot_product2 = (dot_prod)**2

            distances.append(dist)  # Append distance
            dot_prods.append(dot_prod)  # Append raw dot product
            dot_products.append(dot_product2)  # Append normalized dot product squared

    # Convert lists to numpy arrays for further processing
    return np.array(distances), np.array(dot_products), np.array(dot_prods)


if __name__ == "__main__":
    # Define datasets with their respective file paths, scaling factors, and colors for plotting
    datasets = {
        "0.7% Agar (10x)": [
            {
                "file": "Sem2/micro_analysis/0.7/12.02.25/filtered_tables/10xBF_0.csv",
                "scale": 0.6510001,  # Pixel-to-micron conversion for 10x magnification
                "color": "blue"  # Color for plotting
            }
        ],
        "1.7% Agar (20x)": [
            {
                "file": "Sem2/micro_analysis/1.7/12.02.25/filtered_tables/20xPC_0.csv",
                "scale": 0.3236001,  # Pixel-to-micron conversion for 20x magnification
                "color": "green"  # Color for plotting
            }
        ]
    }

    # Define the output folder for saving processed data
    output_folder = "average_data"
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Create a single subplot for Average Dot Product Squared
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))  # Set the figure size

    # Loop through each agar concentration
    for label, dataset_list in datasets.items():
        for idx, info in enumerate(dataset_list):  # Loop through each dataset for the same agar concentration
            file_path = info["file"]  # Path to the CSV file
            pixel_to_micron = info["scale"]  # Scaling factor for converting pixels to microns

            # Load the CSV data into a DataFrame
            data = pd.read_csv(file_path)
            positions = data[['X', 'Y']].to_numpy()  # Extract positions (X, Y)
            displacements = data[['DX', 'DY']].to_numpy()  # Extract displacements (DX, DY)

            # Compute distances and dot products
            distances, dot_products, dot_prods = compute_dot_product_and_distance(positions, displacements)
            micron_distances = distances * pixel_to_micron  # Convert distances to microns

            # Determine the maximum distance for binning
            max_distance = np.max(micron_distances)
            cutoff_distance = max_distance

            # Define binning parameters
            num_bins = 100  # Number of bins
            bin_edges = np.linspace(0, cutoff_distance, num_bins + 1)  # Bin edges
            bin_indices = np.digitize(micron_distances, bin_edges) - 1  # Assign distances to bins

            # Compute average dot product in each bin
            avg_dot_products = []  # List to store average dot products
            bin_centers = []  # List to store bin centers

            # Find the indices of elements in bin 0
            bin_0_indices = (bin_indices == 0)

            # Extract the dot_products and dot_prods corresponding to bin 0
            dot_products_bin_0 = dot_products[bin_0_indices]
            dot_prods_bin_0 = dot_prods[bin_0_indices]

            # Ensure bin 0 is not empty
            if len(dot_products_bin_0) == 0 or len(dot_prods_bin_0) == 0:
                print(f"Warning: Bin 0 is empty for {label} (Dataset {idx + 1})!")
                continue

            # Compute the mean of dot_products and dot_prods for bin 0
            mean_dot_products_bin_0 = np.mean(dot_products_bin_0)
            mean_dot_prods_bin_0 = np.mean(dot_prods_bin_0)

            # Use the combined mean for normalization
            for i in range(num_bins):
                in_bin = (bin_indices == i)  # Mask for elements in the current bin
                if np.any(in_bin):  # Only process non-empty bins
                    avg_dot_prods2 = (np.mean(dot_prods[in_bin]))**2
                    avg_dot_products.append(((np.mean(dot_products[in_bin])) - avg_dot_prods2) / (mean_dot_products_bin_0 - avg_dot_prods2))
                    bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)  # Compute bin center

            # Convert to NumPy arrays for calculations
            bin_centers = np.array(bin_centers)
            avg_dot_products = np.array(avg_dot_products)

            # Save bin centers and average dot products to a CSV file in the "average_data" folder
            output_data = pd.DataFrame({
                "Bin Center (µm)": bin_centers,
                "Average Dot Product": avg_dot_products
            })
            # Add a unique identifier (e.g., index) to the filename
            output_file = os.path.join(output_folder, f"{label.replace(' ', '_').replace('%', 'pct')}_dataset_{idx + 1}_renormalised.csv")
            output_data.to_csv(output_file, index=False)
            print(f"Saved data for {label} (Dataset {idx + 1}) to {output_file}")

            # Plot Average Dot Product
            axs.plot(bin_centers, avg_dot_products, label=f"{label} (Dataset {idx + 1})", color=info["color"], linewidth=1.5, linestyle='-')

    # Graph settings for Average Dot Product
    axs.set_xlabel(r"Distance ($\mu m$)")  # Label for x-axis
    axs.set_ylabel("Average Dot Product Squared")  # Label for y-axis
    axs.set_title("Average Dot Product Squared vs Distance for Different Agar Concentrations")  # Plot title
    axs.legend()  # Add legend
    axs.grid(True, linestyle="--", alpha=0.5)  # Add grid with dashed lines

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plot