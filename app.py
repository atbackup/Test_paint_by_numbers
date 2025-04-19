import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
from skimage.filters import sobel  # Import Sobel filter for edge detection

st.set_page_config(page_title="Paint by Numbers App")

st.title("🎨 Paint by Numbers App")
st.markdown("Upload your image and we’ll turn it into a Paint-by-Numbers template.")

# Upload the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    # Let the user choose number of color levels
    num_colors = st.slider("Choose number of color levels", min_value=2, max_value=10, value=5)

    # Resize for faster processing
    image = image.resize((256, 256))
    
    # Convert to grayscale
    gray_image = ImageOps.grayscale(image)
    st.image(gray_image, caption="Grayscale Image", use_container_width=True)

    # Convert to numpy and apply quantization using slider value
    img_array = np.array(gray_image)
    step = 256 // num_colors
    quantized = (img_array // step) * step  # Reduce grayscale levels
    bw_img = Image.fromarray(quantized)

    # Edge detection to create outlines (using Sobel filter)
    edges = sobel(img_array)  # Get edges using Sobel filter
    edges = (edges * 255).astype(np.uint8)  # Convert to 0-255 range
    edge_img = Image.fromarray(edges)

    # Display edges and paint-by-numbers image with outlines
    st.image(edge_img, caption="Outline (Edge Detection)", use_container_width=True)
    st.image(bw_img, caption=f"Paint-by-Numbers Style with {num_colors} Levels", use_container_width=True)

    # Download link for the final image with outlines
    buf = BytesIO()
    bw_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Paint-by-Numbers Image with Outlines", byte_im, file_name="paint_by_numbers_with_outlines.png", mime="image/png")
