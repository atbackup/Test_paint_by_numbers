import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from io import BytesIO
from sklearn.cluster import KMeans  # Import KMeans for color quantization
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

    # Enhance the edge contrast
    edges_enhanced = ImageEnhance.Contrast(edges_image).enhance(3)  # Boost contrast
    edges_enhanced = edges_enhanced.convert('1')  # Convert to binary black & white

    # Create a white background image
    white_bg = Image.new('L', edges_enhanced.size, 255)  # 255 = white

    # Composite: draw black edges on white background
    final_overlay = Image.composite(Image.new('L', edges_enhanced.size, 0), white_bg, edges_enhanced)

    # Display the enhanced edge image with white background
    st.image(final_overlay, caption="Paint-by-Numbers with Enhanced Edges on White Background", use_container_width=True)

    # Download link for the paint-by-numbers image
    buf = BytesIO()
    final_overlay.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Paint-by-Numbers Image", byte_im, file_name="paint_by_numbers.png", mime="image/png")
