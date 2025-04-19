import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
from sklearn.cluster import KMeans  # Import KMeans for color quantization

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

    # Convert the quantized image to a grayscale version for the outlines
    gray_image = ImageOps.grayscale(quantized_image)
    st.image(gray_image, caption="Grayscale Image for Outlines", use_container_width=True)

    # Download link for the paint-by-numbers image
    buf = BytesIO()
    quantized_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Paint-by-Numbers Image", byte_im, file_name="paint_by_numbers.png", mime="image/png")
