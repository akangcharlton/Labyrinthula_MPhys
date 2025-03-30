import pandas as pd  # For handling CSV data
import numpy as np  # For numerical operations
import cv2  # For image processing
import glob  # For file pattern matching
import os  # For file and directory operations
import matplotlib.pyplot as plt  # For plotting and visualizing data

# Define folder paths
image_folder = "0.7/05.02.25/images/"  # Folder containing grayscale image files
vector_folder = "0.7/05.02.25/filtered_tables/"  # Folder containing filtered vector data in CSV files

# Get all filtered CSV files in the specified folder
filtered_csv_files = glob.glob(os.path.join(vector_folder, "*.csv"))

# Iterate over each filtered CSV file
for csv_file in filtered_csv_files:
    # Extract the filename without the extension (e.g., "file1")
    file_name = os.path.splitext(os.path.basename(csv_file))[0]

    # Construct the corresponding image file path
    image_file = os.path.join(image_folder, f"{file_name}.tif")

    # Check if the image file exists
    if not os.path.exists(image_file):
        # If the image file is missing, print a message and skip this CSV file
        print(f"Image file {image_file} not found. Skipping {csv_file}.")
        continue

    # Load the background image in grayscale mode
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        # If the image could not be loaded, print an error message and skip this CSV file
        print(f"Error loading image {image_file}. Skipping {csv_file}.")
        continue

    # Convert the grayscale image to RGB for better visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Load the filtered vector data from the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Ensure the required columns ('X', 'Y', 'Orientation') exist in the DataFrame
    if not {'X', 'Y', 'Orientation'}.issubset(df.columns):
        # If any required column is missing, print a message and skip this CSV file
        print(f"Missing required columns in {csv_file}. Skipping.")
        continue

    # Extract the X, Y coordinates and orientation values from the DataFrame
    x = df["X"]  # X-coordinates of vectors
    y = df["Y"]  # Y-coordinates of vectors
    angles = df["Orientation"]  # Orientation of vectors (assumed to be in degrees)

    # Convert orientation angles to unit vectors for plotting
    u = np.cos(np.radians(angles))  # X-component of the vector
    v = -np.sin(np.radians(angles))  # Y-component of the vector (negative for correct orientation)

    # Create a plot to overlay vectors on the image
    fig, ax = plt.subplots(figsize=(10, 10))  # Set the figure size
    ax.imshow(image_rgb, cmap="gray", origin="upper")  # Display the background image

    # Plot the vectors using a quiver plot
    ax.quiver(
        x, y, u, v, angles="xy", scale_units="xy", scale=0.05, color="yellow",
        width=0.001, headwidth=0, headlength=0, headaxislength=0
    )

    # Adjust the plot limits to match the image dimensions
    ax.set_xlim(0, image.shape[1])  # Set x-axis limits to the image width
    ax.set_ylim(image.shape[0], 0)  # Set y-axis limits to the image height (invert y-axis)
    ax.set_title(f"Vector Overlay: {file_name}")  # Set the plot title

    # Define the output folder for saving overlay images
    output_folder = f"1.7/12.02.25/overlayed_images/"
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

    # Construct the output file path for the overlay image
    output_file = os.path.join(output_folder, f"{file_name}_overlay.tif")

    # Save the overlay image to the output file
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()  # Close the figure to free memory

    # Print a message indicating that the overlay image has been saved
    print(f"Overlay saved: {output_file}")
