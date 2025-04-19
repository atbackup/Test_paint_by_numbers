import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from io import BytesIO

st.set_page_config(page_title="Paint by Numbers App")

st.title("🎨 Paint by Numbers App")
st.markdown("Upload your image and we’ll turn it into a Paint-by-Numbers template.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def quantize_image(image, n_colors):
    # Convert the image to a NumPy array
    img_array = np.array(image)
    
    # Reshape the image to a 2D array of pixels
    pixels = img_array.reshape(-1, 3)
    
    # Apply KMeans clustering for color quantization
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get the quantized colors
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(int)
    
    # Reshape the pixels back to the original image shape
    quantized_img = quantized_pixels.reshape(img_array.shape)
    
    return Image.fromarray(quantized_img.astype(np.uint8))

if uploaded_file is not None:
    # Load and display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    # Let the user choose the number of color levels
    num_colors = st.slider("Choose number of color levels", min_value=2, max_value=10, value=5)

    # Resize for speed
    image = image.resize((256, 256))

    # Apply color quantization
    quantized_image = quantize_image(image, num_colors)
    st.image(quantized_image, caption="Paint-by-Numbers Style", use_container_width=True)

    # Download link
    buf = BytesIO()
    quantized_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Paint-by-Numbers Image", byte_im, file_name="paint_by_numbers.png", mime="image/png")
