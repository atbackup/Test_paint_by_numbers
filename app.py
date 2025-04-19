import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Paint by Numbers App")

st.title("🎨 Paint by Numbers App")
st.markdown("Upload your image and we’ll turn it into a Paint-by-Numbers template.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    # Resize for speed
    image = image.resize((256, 256))
    
    # Convert to grayscale
    gray_image = ImageOps.grayscale(image)
    st.image(gray_image, caption="Grayscale Image", use_container_width=True)

    # Convert to numpy and quantize to 5 levels
    img_array = np.array(gray_image)
    quantized = (img_array // 51) * 51  # reduce to ~5 levels
    bw_img = Image.fromarray(quantized)
    st.image(bw_img, caption="Paint-by-Numbers Style", use_container_width=True)

    # Download link
    buf = BytesIO()
    bw_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Paint-by-Numbers Image", byte_im, file_name="paint_by_numbers.png", mime="image/png")
