import streamlit as st
from PIL import Image
import os
import random
import numpy as np
import io
import zipfile
import shutil

# Function for JPEG compression using Pillow (PIL)
def compress_image(image, compression_level):
    # Convert image to RGB mode (necessary for JPEG)
    if image.mode != 'L':
        image = image.convert("L")
    
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

    return best_compression_level, best_mse

# Function to calculate Mean Squared Error (MSE) between two grayscale images
def calculate_mse(image1, image2):
    arr1 = np.array(image1.convert("L"))  # Convert to grayscale
    arr2 = np.array(image2.convert("L"))  # Convert to grayscale
    mse = np.mean((arr1 - arr2) ** 2)
    return mse

# Function to calculate Peak Signal-to-Noise Ratio (PSNR) between two grayscale images
def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Process images from the input folder and save compressed images to the output folder
def process_images(input_folder, output_folder, target_size, max_iterations=50, initial_compression_level=80):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Evaluation metrics for each image
    evaluation_metrics = []

    # Process each image in the input folder
    for file in input_folder:
        image = Image.open(file)

        # Perform stem cell algorithm for optimization with the target size
        best_compression_level, best_mse = stem_cell_algorithm(image, target_size, max_iterations, initial_compression_level)

        # Perform compression using the optimized compression level
        compressed_image, compressed_size, _ = compress_image(image, best_compression_level)

        # Save the compressed image to the output folder
        compressed_file_path = os.path.join(output_folder, file.name)
        compressed_image.save(compressed_file_path, format='JPEG')

        # Calculate evaluation metrics for each image and store in the list
        psnr = calculate_psnr(image, compressed_image)
        mse = calculate_mse(image, compressed_image)
        evaluation_metrics.append((file.name, psnr, mse))

    return evaluation_metrics

# Streamlit web application
def main():
    st.title("Medical Image Compression using Stem Cell Algorithm")
    st.write("Upload a lung image or a folder of images, and we'll optimize the compression level for you!")
    st.write("Project by Lois juwonlo Ajani--19/47cs/01023 and 20D/47CS/01260--Olaniran Samuel Adedayo")

    # File uploader for single image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # File uploader for folder of images
    input_folder = st.file_uploader("Upload a folder of images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_file is not None:
        # Display the original image
        st.subheader("Original Image")
        st.image(uploaded_file, caption="Original Image", use_column_width=True)

        # Compression level slider
        initial_compression_level = st.slider("Select Initial Compression Level", min_value=1, max_value=100, value=80)

        # Perform stem cell algorithm for optimization with a target size of 80% of the original size
        image = Image.open(uploaded_file)
        target_size = 0.8 * len(uploaded_file.getvalue())
        best_compression_level, best_mse = stem_cell_algorithm(image, target_size, max_iterations=50, initial_compression_level=initial_compression_level)

        # Perform compression using the optimized compression level
        compressed_image, compressed_size, compressed_buffer = compress_image(image, best_compression_level)

        # Display the compressed image
        st.subheader("Compressed Image")
        st.image(compressed_image, caption="Compressed Image", use_column_width=True)
        st.write(f"Compressed Image Size: {compressed_size} bytes")

        # Evaluation Metrics
        psnr = calculate_psnr(image, compressed_image)
        mse = calculate_mse(image, compressed_image)

        st.subheader("Evaluation Metrics")
        st.write(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")

        # Visualization of Metrics as Graph
        if st.button("Show Metrics Graph"):
            metrics_data = {
                "PSNR": [calculate_psnr(image, compress_image(image, compression_level)[0]) for compression_level in range(1, 101)],
                "MSE": [calculate_mse(image, compress_image(image, compression_level)[0]) for compression_level in range(1, 101)]
            }
            st.line_chart(metrics_data)

        # Provide a download link for the compressed image
        st.download_button("Download Compressed Image", data=compressed_buffer, file_name="compressed_image.jpg")
        st.write("note optimization works perfectly on medical images only")

    if input_folder is not None:
        # Compress images in the uploaded folder and save to another folder
        st.subheader("Compress Images in a Folder")
        output_folder = st.text_input("Enter the path of the output folder:")
        if st.button("Compress Images"):
            try:
                output_folder = os.path.normpath(output_folder)
                target_size = 0.8 * 320 * 1024  # Target size of 80% of 320KB
                evaluation_metrics = process_images(input_folder, output_folder, target_size)
                st.success("Image compression completed!")

                # Display evaluation metrics for each image
                st.subheader("Evaluation Metrics for Compressed Images")
                for filename, psnr, mse in evaluation_metrics:
                    st.write(f"Image: {filename}, PSNR: {psnr:.2f} dB, MSE: {mse:.2f}")
                
                # Create a ZIP file containing all the compressed images
                zip_file_path = os.path.join("./", "compressed_images.zip")
                
                with zipfile.ZipFile(zip_file_path, "w") as zip_file:
                    for filename in os.listdir(output_folder):
                        file_path = os.path.join(output_folder, filename)
                        with open(file_path, "rb") as file:
                            zip_file.writestr(os.path.basename(file_path), file.read())

                # Provide a download link for the ZIP file
                st.download_button("Download All Compressed Images", data=open(zip_file_path, "rb").read(), file_name="compressed_images.zip")
                shutil.rmtree(output_folder)

                
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
