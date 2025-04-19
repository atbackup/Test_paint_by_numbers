import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
from sklearn.cluster import KMeans
from skimage import filters
import matplotlib.pyplot as plt

st.set_page_config(page_title="Paint by Numbers App")

st.title("🎨 Paint by Numbers App")
st.markdown("Upload your image and we’ll turn it into a Paint-by-Numbers template.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    # Let the user choose the number of color levels
    num_colors = st.slider("Choose number of color levels", min_value=2, max_value=10, value=5)

    # Resize for speed
    image = image.resize((256, 256))

    # Convert the image to a numpy array
    img_array = np.array(image)

    # Reshape the image into a 2D array of pixels
    img_reshaped = img_array.reshape((-1, 3))

    # Perform KMeans clustering on the image's pixels to reduce colors
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(img_reshaped)
    
    # Get the cluster centers (the colors to be used in the quantized image)
    new_colors = kmeans.cluster_centers_.astype(int)

    # Replace each pixel with its corresponding cluster center (the color)
    quantized_img_reshaped = new_colors[kmeans.labels_]
    quantized_img = quantized_img_reshaped.reshape(img_array.shape)

    # Convert the quantized image back to a PIL image
    quantized_image = Image.fromarray(quantized_img.astype(np.uint8))

    # Display the quantized image (paint-by-numbers style)
    st.image(quantized_image, caption="Paint-by-Numbers Style", use_container_width=True)

    # Convert to grayscale
    gray_image = ImageOps.grayscale(image)

    # Convert to numpy array
    gray_array = np.array(gray_image)

    # Apply Sobel filter to detect edges
    edges = filters.sobel(gray_array)

    # Normalize edge values (edges are between 0 and 1)
    edges = (edges * 255).astype(np.uint8)

    # Convert back to an image
    edges_image = Image.fromarray(edges)

    # Display grayscale and edge-detected images
    st.image(gray_image, caption="Grayscale Image", use_container_width=True)
    st.image(edges_image, caption="Edge Detected Image", use_container_width=True)

    # Overlay the edges on the paint-by-numbers image (quantized_image)
    bw_img_with_edges = Image.composite(quantized_image.convert('L'), edges_image.convert('L'), edges_image)
    st.image(bw_img_with_edges, caption="Paint-by-Numbers with Edges", use_container_width=True)

    # Adding numbers for each region (adjusted for clarity)
    draw = ImageDraw.Draw(bw_img_with_edges)
    font = ImageFont.load_default()  # Use default font or choose a larger one
    width, height = bw_img_with_edges.size
    
    # Loop over each cluster (color region)
    for i in range(num_colors):
        # Find all pixel indices that belong to the current cluster
        cluster_indices = np.where(kmeans.labels_ == i)

        # Check if cluster_indices are valid (ensure that the tuple contains valid indices)
        if len(cluster_indices[0]) > 0:  # Ensure the cluster has valid row and column indices
            row_indices = cluster_indices[0]  # row indices (vertical positions)
            col_indices = cluster_indices[1]  # column indices (horizontal positions)
            
            # Calculate the centroid of the cluster (mean of x and y coordinates of the pixels)
            region_x = np.mean(col_indices)  # x-coordinates (columns)
            region_y = np.mean(row_indices)  # y-coordinates (rows)

            # Add number to the image with a small shadow for contrast
            draw.text((region_x + 1, region_y + 1), str(i + 1), fill=(255, 255, 255), font=font)  # shadow
            draw.text((region_x, region_y), str(i + 1), fill=(0, 0, 0), font=font)  # main text
        else:
            st.write(f"Cluster {i} has no valid pixels")

    # Show the updated image with numbers
    st.image(bw_img_with_edges, caption="Paint-by-Numbers with Numbers", use_container_width=True)

    # Download link for the paint-by-numbers image
    buf = BytesIO()
    bw_img_with_edges.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Paint-by-Numbers Image", byte_im, file_name="paint_by_numbers.png", mime="image/png")
