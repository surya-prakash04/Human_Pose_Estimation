import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Constants for body parts and pose pairs
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
    ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
    ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"],
    ["Nose", "LEye"], ["LEye", "LEar"]
]

# Image dimensions
inWidth = 368
inHeight = 368

@st.cache_resource
def load_model():
    """Load the pose estimation model."""
    static_model_path = "graph_opt (1).pb"  # Replace with your static model path
    return cv2.dnn.readNetFromTensorflow(static_model_path)

def poseDetector(frame, thresh):
    """Detect poses in the given frame."""
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frame_width * point[0]) / out.shape[3]
        y = (frame_height * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thresh else None)

    for pair in POSE_PAIRS:
        part_from, part_to = pair
        id_from, id_to = BODY_PARTS[part_from], BODY_PARTS[part_to]
        if points[id_from] and points[id_to]:
            cv2.line(frame, points[id_from], points[id_to], (0, 255, 0), 3)
            cv2.ellipse(frame, points[id_from], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[id_to], (3 , 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

def load_image(image_file):
    """Load an image from a file buffer or return a demo image."""
    if image_file is not None:
        return np.array(Image.open(image_file))
    else:
        demo_image_path = "enterprise_2.jpg"  # Replace with your demo image path
        return np.array(Image.open(demo_image_path))

# Streamlit UI with background
st.markdown(
    """
    <style>
    .main {
        background: url('https://source.unsplash.com/1920x1080/?dark-clouds') no-repeat center center fixed;
        background-size: cover;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌩️ Human Pose Estimation with Style 🌩️")
st.markdown("This application demonstrates human pose estimation using OpenCV with a stylish dark-cloud background. Upload an image or use the demo image provided.")

# Sidebar for threshold settings
thresh_value = st.sidebar.slider('Threshold for detecting key points', min_value=0, value=20, max_value=100, step=5)
thresh = thresh_value / 100

# Load the model
try:
    net = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# File uploader
img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load the image
image = load_image(img_file_buffer)

# Display original image
st.subheader('Original Image')
st.image(image, caption="Original Image", use_container_width=True)

# Process the image
with st.spinner('Processing image...'):
    output = poseDetector(image, thresh)

# Display estimated pose
st.subheader('Estimated Pose')
st.image(output, caption="Pose Estimation Result", use_container_width=True)

# Option to download the processed image
st.sidebar.download_button(
    label="Download Pose Image",
    data=cv2.imencode('.jpg', output)[1].tobytes(),
    file_name='pose_estimation.jpg',
    mime='image/jpeg'
)
