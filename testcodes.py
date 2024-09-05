import imageio
import cv2

def save_video_from_frames(frames, output_path, fps=30):
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

# Example usage
cam = cv2.VideoCapture(0) 




frames = [numpy_array1, numpy_array2, numpy_array3]  # Replace with your list of NumPy arrays
save_video_from_frames(frames, 'output_video.mp4')
