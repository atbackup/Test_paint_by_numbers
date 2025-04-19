import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Streamlit app title
st.title("🎨 Paint by Numbers with Edge Enhancement")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    image_array = np.array(img)

    # Number of colors (clusters) you want to use for color quantization
    num_colors = st.slider("Choose number of color levels", min_value=2, max_value=10, value=5)

    # Reshaping the image to a 2D array of pixels (each pixel as a row of RGB values)
    reshaped_image = image_array.reshape((-1, 3))

    # Applying KMeans clustering to quantize the colors
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(reshaped_image)

    # Assign each pixel to the nearest cluster
    labels = kmeans.labels_

    # Convert labels back to the image shape
    segmented_image = kmeans.cluster_centers_[labels].reshape(image_array.shape)

    # Convert segmented image to a PIL Image
    seg_img = Image.fromarray(segmented_image.astype('uint8'))

    # Create a drawing object
    draw = ImageDraw.Draw(seg_img)

    # Set up font for numbers (ensure this path is correct or use a default one)
    font = ImageFont.load_default()  # You can load a specific font if necessary

    # Loop through each cluster and draw numbers
    for i in range(num_colors):
        # Get the indices of pixels belonging to this cluster
        cluster_indices = np.where(kmeans.labels_ == i)

        # Ensure that the cluster has valid pixels
        if len(cluster_indices[0]) > 0:  # Check if the cluster has valid pixels
            row_indices = cluster_indices[0]  # row indices (vertical positions)
            col_indices = cluster_indices[1]  # column indices (horizontal positions)

            # Calculate the centroid of the cluster (mean of x and y coordinates of the pixels)
            region_x = np.mean(col_indices)  # x-coordinates (columns)
            region_y = np.mean(row_indices)  # y-coordinates (rows)

            # Add number to the image with a small shadow for contrast
            draw.text((region_x + 1, region_y + 1), str(i + 1), fill=(255, 255, 255), font=font)  # shadow
            draw.text((region_x, region_y), str(i + 1), fill=(0, 0, 0), font=font)  # main text
        else:
            st.write(f"Cluster {i} has no valid pixels.")

    # Show the segmented image with numbers on the Streamlit app
    st.image(seg_img, caption="Paint-by-Numbers with Numbers", use_column_width=True)

    # Allow user to download the image (optional)
    img_save_path = "/tmp/paint_by_numbers_image.png"
    seg_img.save(img_save_path)
    st.download_button(label="Download Image", data=open(img_save_path, "rb"), file_name="paint_by_numbers_image.png")
