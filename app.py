import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
from sklearn.cluster import KMeans
import cv2

st.set_page_config(page_title="Paint by Numbers App")

st.title("🎨 Paint by Numbers App")
st.markdown("Upload your image and we’ll turn it into a Paint-by-Numbers template with clean outlines. Add colors to paint!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    # Choose number of color levels
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
    blurred = cv2.GaussianBlur(gray_quantized, (3, 3), 0)

    # Use Canny edge detection for sharp outlines
    edges = cv2.Canny(blurred, 30, 100)

    # Invert edges so they appear black on white
    edges_inv = cv2.bitwise_not(edges)

    # Convert single channel to 3-channel RGB
    paint_by_numbers_img = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2RGB)

    # Add bold blue numbers to the regions
    contours, _ = cv2.findContours(edges_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    numbered_img = paint_by_numbers_img.copy()
    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour) > 100:  # Avoid very small contours
            # Find the center of the contour
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                # Add the number to the region with blue color and bold font
                cv2.putText(numbered_img, 
                            str(idx + 1),  # Number to display
                            (cx - 10, cy + 10),  # Position of the number (center of the contour)
                            cv2.FONT_HERSHEY_SIMPLEX,  # Font
                            1,  # Font scale
                            (255, 0, 0),  # Blue color (BGR format)
                            3,  # Thickness (boldness)
                            cv2.LINE_AA)  # Anti-aliasing for smoother text

    # Display result
    st.image(numbered_img, caption="🖼️ Final Paint-by-Numbers Template with Numbers", use_container_width=True)

    # Download button
    result_image = Image.fromarray(numbered_img)
    buf = BytesIO()
    result_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Paint-by-Numbers Image", byte_im, file_name="paint_by_numbers_outlines_with_numbers.png", mime="image/png")

    # User painting functionality (basic version):
    st.markdown("### Paint the Numbers! (Click on a region to fill with color)")
    color = st.color_picker("Pick a color", "#000000")  # Default color is black

    # Placeholder for painting action (this would be enhanced with more complex interaction logic)
    st.write("Click on the regions to apply your chosen color.")
