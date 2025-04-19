import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
from sklearn.cluster import KMeans
import cv2

st.set_page_config(page_title="Paint by Numbers App")

st.title("🎨 Paint by Numbers App")
st.markdown("Upload your image and we’ll turn it into a Paint-by-Numbers template with clean outlines.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    # Choose number of color areas
    num_colors = st.slider("Choose number of color areas", 2, 10, 5)

    # Resize image for performance
    image = image.resize((256, 256))
    img_array = np.array(image)

    # Color quantization using KMeans
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)
    new_colors = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_

    # Reconstruct quantized image
    quantized = new_colors[labels].reshape(img_array.shape)

    # Convert to grayscale for edge detection
    gray_quantized = cv2.cvtColor(quantized.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur (optional) to reduce noise
    blurred = cv2.GaussianBlur(gray_quantized, (5, 5), 1)

    # Use Canny edge detection for sharp outlines
    edges = cv2.Canny(blurred, 50, 150)  # Adjust these values for sharper edges

    # Invert edges so they appear black on white
    edges_inv = cv2.bitwise_not(edges)

    # Convert single channel to 3-channel RGB (white background, black outlines)
    paint_by_numbers_img = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2RGB)

    # Set the background to white (just in case it's not pure white)
    paint_by_numbers_img[paint_by_numbers_img == 0] = 255

    # Display result
    st.image(paint_by_numbers_img, caption="🖼️ Final Paint-by-Numbers Outline", use_container_width=True)

    # Download button
    result_image = Image.fromarray(paint_by_numbers_img)
    buf = BytesIO()
    result_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Paint-by-Numbers Image", byte_im, file_name="paint_by_numbers_outlines.png", mime="image/png")
