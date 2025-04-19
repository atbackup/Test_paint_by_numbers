import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
from sklearn.cluster import KMeans  # Import KMeans for color quantization
import cv2  # OpenCV for better edge detection
from matplotlib import pyplot as plt
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="Paint by Numbers App")

st.title("🎨 Paint by Numbers App")
st.markdown("Upload your image and we’ll turn it into a Paint-by-Numbers template.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    num_colors = st.slider("Choose number of color levels", min_value=2, max_value=10, value=5)

    image = image.resize((256, 256))
    img_array = np.array(image)
    img_reshaped = img_array.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(img_reshaped)
    labels = kmeans.labels_.reshape(img_array.shape[:2])
    cluster_centers = kmeans.cluster_centers_.astype(np.uint8)

    quantized_img = np.zeros_like(img_array)
    for i in range(num_colors):
        quantized_img[labels == i] = cluster_centers[i]

    quantized_image = Image.fromarray(quantized_img)
    st.image(quantized_image, caption="Paint-by-Numbers Style", use_container_width=True)

    gray = cv2.cvtColor(quantized_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edges_inv = cv2.bitwise_not(edges)
    white_bg = np.ones_like(edges_inv) * 255
    outline_img = cv2.bitwise_and(white_bg, white_bg, mask=edges_inv)
    outline_image = Image.fromarray(outline_img)

    st.image(outline_image, caption="Outlined Image (Black lines on white background)", use_container_width=True)

    # Create a new image with numbers drawn in the center of each region
    numbered_img = outline_image.convert("RGB")
    draw = ImageDraw.Draw(numbered_img)
    font = ImageFont.load_default()

    for i in range(num_colors):
        mask = (labels == i)
        coords = np.column_stack(np.where(mask))
        if coords.size > 0:
            y, x = coords.mean(axis=0).astype(int)
            draw.text((x, y), str(i + 1), fill=(0, 0, 0), font=font)

    st.image(numbered_img, caption="Numbered Paint-by-Numbers Template", use_container_width=True)

    # Generate the color legend image
    legend_img = Image.new("RGB", (300, num_colors * 40), color="white")
    draw_legend = ImageDraw.Draw(legend_img)
    for i in range(num_colors):
        color = tuple(cluster_centers[i])
        draw_legend.rectangle([10, i * 40 + 5, 50, i * 40 + 35], fill=color)
        draw_legend.text((60, i * 40 + 10), f"{i + 1}", fill="black", font=font)

    st.image(legend_img, caption="Color Legend", use_container_width=False)

    # Save PDF with numbered image and legend
    with tempfile.TemporaryDirectory() as tmpdir:
        numbered_img_path = os.path.join(tmpdir, "numbered.png")
        legend_img_path = os.path.join(tmpdir, "legend.png")
        pdf_path = os.path.join(tmpdir, "paint_by_numbers.pdf")

        numbered_img.save(numbered_img_path)
        legend_img.save(legend_img_path)

        pdf = FPDF()
        pdf.add_page()
        pdf.image(numbered_img_path, x=10, y=10, w=180)
        pdf.add_page()
        pdf.image(legend_img_path, x=10, y=10, w=100)
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("Download Paint-by-Numbers PDF", f.read(), file_name="paint_by_numbers.pdf", mime="application/pdf")
