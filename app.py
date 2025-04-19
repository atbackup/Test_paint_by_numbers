import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
from sklearn.cluster import KMeans
import cv2  # OpenCV for edge detection

st.set_page_config(page_title="Paint by Numbers App")

st.title("🎨 Paint by Numbers App")
st.markdown("Upload your image and we’ll turn it into a Paint-by-Numbers template.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    # Choose color levels
    num_colors = st.slider("Choose number of color levels", min_value=2, max_value=10, value=5)

    # Resize for speed
    image = image.resize((256, 256))

    # Convert to numpy array
    img_array = np.array(image)

    # Color quantization with KMeans
    img_reshaped = img_array.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(img_reshaped)
    new_colors = kmeans.cluster_centers_.astype(int)
    quantized_img_reshaped = new_colors[kmeans.labels_]
    quantized_img = quantized_img_reshaped.reshape(img_array.shape)
    quantized_image = Image.fromarray(quantized_img.astype(np.uint8))
    st.image(quantized_image, caption="Paint-by-Numbers Style", use_container_width=True)

    # Use OpenCV for edge detection (Canny)
    gray_for_edges = cv2.cvtColor(quantized_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_for_edges, threshold1=50, threshold2=150)

    # Invert edges (white background, black lines → we want black on white)
    edges_inv = cv2.bitwise_not(edges)

    # Convert edge mask to 3 channels and overlay on white background
    white_bg = np.ones_like(img_array, dtype=np.uint8) * 255
    edges_3ch = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2RGB)
    outlined_img = cv2.bitwise_and(white_bg, edges_3ch)

    # Merge with quantized image: blend edges into the quantized image
    outline_mask = edges > 0
    final_image = quantized_img.copy()
    final_image[outline_mask] = 0  # black lines

    # Convert final image to PIL and display
    final_pil = Image.fromarray(final_image.astype(np.uint8))
    st.image(final_pil, caption="Final Paint-by-Numbers with Outlines", use_container_width=True)

    # Download
    buf = BytesIO()
    final_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Paint-by-Numbers Image", byte_im, file_name="paint_by_numbers.png", mime="image/png")
