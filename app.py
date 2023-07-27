import streamlit as st
from PIL import Image
import io
import os
import random
import numpy as np

# Function for JPEG compression using Pillow (PIL)
def compress_image(image, compression_level):
    # Convert image to RGB mode (necessary for JPEG)
    image = image.convert("RGB")
    
    # Save the compressed image to a bytes buffer in JPEG format with lower quality
    output_buffer = io.BytesIO()
    image.save(output_buffer, format='JPEG', quality=compression_level)
    
    # Get the compressed image size
    compressed_size = output_buffer.getbuffer().nbytes
    
    # Seek to the beginning of the buffer for downloading
    output_buffer.seek(0)
    
    # Open and return the compressed image
    return Image.open(output_buffer), compressed_size, output_buffer

# Simple Stem Cell Algorithm for actual optimization
def stem_cell_algorithm(image, target_size, max_iterations=100, initial_compression_level=80):
    best_compression_level = initial_compression_level
    best_mse = float('inf')

    for i in range(max_iterations):
        compression_level = best_compression_level + random.randint(-10, 10)
        compressed_image, compressed_size, _ = compress_image(image, compression_level)

        # Evaluate fitness using Mean Squared Error (MSE)
        mse = calculate_mse(image, compressed_image)

        # Check if compressed image size is within the target size range
        if compressed_size <= target_size:
            # Update best result
            if mse < best_mse:
                best_mse = mse
                best_compression_level = compression_level

    return best_compression_level

# Function to calculate Mean Squared Error (MSE) between two images
def calculate_mse(image1, image2):
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    mse = np.mean((arr1 - arr2) ** 2)
    return mse

# Streamlit web application
def main():
    st.title("Medical Image Compression using Stem Cell Algorithm")
    st.write("Upload a lung image, and we'll optimize the compression level for you!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the original image
        st.image(uploaded_file, caption="Original Image", use_column_width=True)

        # Compression level slider
        initial_compression_level = st.slider("Select Initial Compression Level", min_value=1, max_value=100, value=80)

        # Perform stem cell algorithm for optimization with a target size of 80% of the original size
        image = Image.open(uploaded_file)
        target_size = 0.8 * len(uploaded_file.getvalue())
        best_compression_level = stem_cell_algorithm(image, target_size, max_iterations=50, initial_compression_level=initial_compression_level)

        # Perform compression using the optimized compression level
        compressed_image, compressed_size, compressed_buffer = compress_image(image, best_compression_level)

        # Display the compressed image
        st.image(compressed_image, caption="Compressed Image", use_column_width=True)
        st.write(f"Compressed Image Size: {compressed_size} bytes")

        # Provide a download link for the compressed image
        st.download_button("Download Compressed Image", data=compressed_buffer, file_name="compressed_image.jpg")
main()



# Function to build a Huffman tree given a frequency table
def build_huffman_tree(frequency_table):
    # Convert the frequency table to an array of nodes
    nodes = [{"value": value, "frequency": frequency} for value, frequency in frequency_table.items()]

    # Build the Huffman tree by merging nodes with the lowest frequency
    while len(nodes) > 1:
        nodes.sort(key=lambda x: x["frequency"])
        left = nodes.pop(0)
        right = nodes.pop(0)
        merged_node = {"value": None, "frequency": left["frequency"] + right["frequency"], "left": left, "right": right}
        nodes.append(merged_node)

    # Return the root of the Huffman tree
    return nodes[0]

# Function to perform entropy encoding using Huffman coding
def entropy_encode(stem):
    # Create a frequency table for the stem's pixel values
    frequency_table = {}
    for value in stem:
        frequency_table[value] = frequency_table.get(value, 0) + 1

    # Build the Huffman tree using the frequency table
    huffman_tree = build_huffman_tree(frequency_table)
        # Save the compressed image to a bytes buffer in JPEG format with lower quality
    output_buffer = io.BytesIO()
    huffman_tree.save(output_buffer, format='JPEG', quality=2)
    
    # Get the compressed image size
    compressed_size = output_buffer.getbuffer().nbytes
    
    # Seek to the beginning of the buffer for downloading
    output_buffer.seek(0)
    
    # Open and return the compressed image
    return Image.open(output_buffer), compressed_size

# Simple Stem Cell Algorithm (just for illustration, not actual optimization)
def stem_cell_algorithm(image, target_size, max_iterations=100, initial_compression_level=80):
    best_compression_level = initial_compression_level
    best_mse = float('inf')

    for iteration in range(max_iterations):
        compression_level = best_compression_level + random.randint(-10, 10)
        compressed_image, compressed_size = compress_image(image, compression_level)

        # Evaluate fitness using Mean Squared Error (MSE)
        mse = calculate_mse(image, compressed_image)

        # Check if compressed image size is within the target size range
        if compressed_size <= target_size:
            # Update best result
            if mse < best_mse:
                best_mse = mse
                best_compression_level = compression_level

    return best_compression_level

# Function to calculate Mean Squared Error (MSE) between two images
def calculate_mse(image1, image2):
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    mse = np.mean((arr1 - arr2) ** 2)
    return mse
