import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
from sklearn.cluster import KMeans
import cv2  # OpenCV for better edge detection

st.set_page_config(page_title="Paint by Numbers App")

st.title("🎨 Paint by Numbers App")
st.markdown("Upload your image and we’ll turn it into a Paint-by-Numbers template with clean outlines.")

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

    new_colors = kmeans.cluster_centers_.astype(int)
    quantized_img_reshaped = new_colors[kmeans.labels_]
    quantized_img = quantized_img_reshaped.reshape(img_array.shape)
    quantized_image = Image.fromarray(quantized_img.astype(np.uint8))

    st.image(quantized_image, caption="Paint-by-Numbers Style", use_container_width=True)

    # Convert image to grayscale for edge detection
    gray = ImageOps.grayscale(image)
    gray_np = np.array(gray)

    # Use OpenCV Canny for clean edge detection
    edges = cv2.Canny(gray_np, threshold1=50, threshold2=150)  # Tune thresholds as needed

    # Invert edges: edges are black (0), background is white (255)
    inverted_edges = cv2.bitwise_not(edges)

    # Convert to PIL image
