import os
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

# Disable Git checks by the ultralytics library
os.environ["YOLO_GIT"] = "0"


# Load the trained YOLO model
@st.cache_resource
def load_model():
    return YOLO("c_qb(755)epoch120.pt")  # Replace with the path to your updated model


def add_tooltip(frame, x, y, label, bg_color, text_color):
    """
    Adds a tooltip with a label above the specified position in a chat icon style.
    Args:
        frame (np.ndarray): The video frame.
        x (int): X-coordinate for tooltip position.
        y (int): Y-coordinate for tooltip position.
        label (str): The label to display.
        bg_color (tuple): BGR color for the tooltip background.
        text_color (tuple): BGR color for the text.
    """
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_w, text_h = text_size
    padding = 10
    arrow_height = 10
    corner_radius = 5

    # Define coordinates for the tooltip box
    bg_x1 = x - text_w // 2 - padding
    bg_y1 = y - text_h - padding - arrow_height
    bg_x2 = x + text_w // 2 + padding
    bg_y2 = y - arrow_height

    # Draw the rounded rectangle for the tooltip
    tooltip_box = [
        (bg_x1 + corner_radius, bg_y1),
        (bg_x2 - corner_radius, bg_y1),
        (bg_x2, bg_y1 + corner_radius),
        (bg_x2, bg_y2 - corner_radius),
        (bg_x2 - corner_radius, bg_y2),
        (bg_x1 + corner_radius, bg_y2),
        (bg_x1, bg_y2 - corner_radius),
        (bg_x1, bg_y1 + corner_radius)
    ]
    tooltip_box = np.array([tooltip_box], dtype=np.int32)
    cv2.fillPoly(frame, [tooltip_box], bg_color)

    # Draw a small triangle (arrow) pointing down
    arrow = np.array([
        [x, y],  # Tip of the arrow
        [x - 7, bg_y2],  # Left corner
        [x + 7, bg_y2]   # Right corner
    ], dtype=np.int32)
    cv2.fillPoly(frame, [arrow], bg_color)

    # Add text label inside the tooltip
    text_x = x - text_w // 2
    text_y = bg_y2 - 6
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )

def process_video(video_path, model):
    """
    Processes the video frame by frame and adds tooltips with labels.
    Args:
        video_path (str): Path to the uploaded video file.
        model: Loaded YOLO model.
    Returns:
        None
    """
    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()  # Placeholder to display video frames
    progress_bar = st.progress(0)  # Progress bar for processing
    # Get total frames for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    # Loop through each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Perform object detection with YOLO
        results = model.predict(frame, conf=0.65, iou=0.5)  # Adjust thresholds as needed
        # Annotate the frame with detection results (Tooltip)
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            cls = result.cls[0]  # Class ID
            conf = result.conf[0]  # Confidence score
            class_name = model.names[int(cls)]  # Get the class name
            # Set label text and colors based on class name
            if "QB" in class_name:
                label = "QB"
                bg_color = (0, 255, 255)  # Yellow background
                text_color = (0, 0, 255)  # Red text
            elif "C" in class_name:
                label = "C"
                bg_color = (255, 0, 0)  # Blue background
                text_color = (255, 255, 255)  # White text
            else:
                label = class_name  # Use class name if not QB or C
                bg_color = (0, 255, 0)  # Green background
                text_color = (0, 0, 0)  # Black text
            # Calculate the center of the detected object
            center_x = (x1 + x2) // 2
            center_y = y1 - 10  # Position tooltip above the bounding box
            # Add tooltip
            add_tooltip(frame, center_x, center_y, label, bg_color, text_color)

        # Convert the frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        # Update progress bar
        processed_frames += 1
        progress = min(processed_frames / total_frames, 1.0)
        progress_bar.progress(progress)
    # Cleanup
    cap.release()
    progress_bar.empty()
    st.success("Video processing complete!")


# Streamlit app interface
def main():
    st.title("American Football Position Formation: Center, QB")
    st.write("Upload a football match video to detect.")

    # File uploader for video input
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        temp_video.close()
        st.write("Processing video...")
        model = load_model()  # Load YOLO model
        process_video(temp_video.name, model)  # Process the video
        # Cleanup temporary file
        os.remove(temp_video.name)


if __name__ == "__main__":
    main()