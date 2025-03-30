import pandas as pd  # For handling CSV data
import numpy as np  # For numerical operations
import cv2  # For image processing
import glob  # For file pattern matching
import os  # For file and directory operations

# Define folder paths (update as needed)
original_data_folder = "Sem1/Analysis/Synthetic_Images/OrientationJ/vector_tables/"  # Folder containing CSV files
mask_folder = "Sem1/Analysis/Synthetic_Images/OrientationJ/masks/"  # Folder containing TIFF mask files

# Get all original CSV files in the specified folder
csv_files = glob.glob(os.path.join(original_data_folder, "*.csv"))

# Iterate over each CSV file
for csv_file in csv_files:
    # Extract the filename without extension (e.g., "Pos001_002")
    file_name = os.path.splitext(os.path.basename(csv_file))[0]
    
    # Construct the corresponding mask file path based on the filename
    mask_file = os.path.join(mask_folder, f"{file_name}.tif")
    
    # Check if the corresponding mask file exists
    if not os.path.exists(mask_file):
        # If the mask file is missing, print a message and skip this CSV file
        print(f"Mask file {mask_file} not found. Skipping {csv_file}.")
        continue

    # Load the original CSV data into a DataFrame
    original_df = pd.read_csv(csv_file)

    # Load the binary mask image in grayscale mode
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    # Check if the mask was loaded successfully
    if mask is None:
        # If the mask could not be loaded, print an error message and skip this CSV file
        print(f"Error loading mask {mask_file}. Skipping {csv_file}.")
        continue

    # Extract the (x, y) coordinates where the mask has a value of 255 (white pixels)
    # OpenCV represents images as (row, col), so we use np.where to find these coordinates
    mask_coords = np.column_stack(np.where(mask == 255))
    
    # Create a DataFrame from the mask coordinates with columns 'Y' and 'X'
    # Note: OpenCV uses (row=y, col=x) convention
    mask_df = pd.DataFrame(mask_coords, columns=['Y', 'X'])

    # Merge the original DataFrame with the mask DataFrame on the 'X' and 'Y' columns
    # Use a left join and add an indicator column to track matches
    filtered_df = original_df.merge(mask_df, on=['X', 'Y'], how='left', indicator=True)
    
    # Filter out rows that matched with the mask (i.e., keep only rows where '_merge' is 'left_only')
    filtered_df = filtered_df[filtered_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Construct the output file path for the filtered data
    output_file = f"Sem1/Analysis/Synthetic_Images/OrientationJ/filtered_tables/{file_name}.csv"
    
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_file, index=False)

    # Print a message indicating that the filtered data has been saved
    print(f"Filtered data saved to {output_file}")
