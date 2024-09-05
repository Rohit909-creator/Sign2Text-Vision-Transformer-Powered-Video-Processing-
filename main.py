import gradio as gr
import time
import cv2
import mediapipe as mp
import numpy as np
import torch
from vit_pytorch import SimpleViT
import json
import imageio
from LLM_ASSIST import *

with open(r"C:\Users\ROHIT FRANCIS\OneDrive\Desktop\AI_Projects\Sign Language Assist\Data.json", "r") as f:
    s = f.read()
    cls_dict = json.loads(s)

print(cls_dict.keys())
print(cls_dict.values())
# print(cls_dict['39'])
# print(f'Classes: {cls_dict.values()}')
chat = Chat()
# exit(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


data_dict = dict()

mp_pose = mp.solutions.pose.Pose()
# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands.Hands()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose.Pose()
    
# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands.Hands()

device = torch.device('cuda')

model = SimpleViT(
        image_size=2048,
        patch_size=128,
        num_classes=351,# 256/32
        dim=1024,
        depth = 8,
        heads = 4,
        mlp_dim=2048,
    )

path = r"C:\Users\ROHIT FRANCIS\Downloads\ViTModel2.pt"

model.load_state_dict(torch.load(path))
model.eval()
model = model.to(device)
print(model.named_modules)


def save_video_from_frames(frames, output_path, fps=30):
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

def convert_to_pose(image):

    blank_image = np.zeros(shape=(430, 640, 3))
        
    results_pose = mp_pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    results_hands = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    if results_pose.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            blank_image,
            results_pose.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
        )

    # Draw hand landmarks and connections
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                blank_image,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS
            )

    return blank_image

def read_and_store(path):
    global frame
    global masked_image
    cap = cv2.VideoCapture(path)
    images = []
    masked_images = []
    while True:
        
        ret, frame = cap.read()

        if not ret:
            break

        masked_image = convert_to_pose(frame)

        if cv2.waitKey(1) == ord('q'):
            break
        frame = cv2.resize(frame, (128,128))
        masked_image = cv2.resize(masked_image, (128,128))
        images.append(frame)
        masked_images.append(masked_image)

    return masked_images, images

def compress_image(imgs_of_a_class, patch_size, exit_patch=56):
    # Initialize the compressed image
    compressed_image = np.zeros((2048, 2048, 3))
    patch_size = 128
    # Initialize patch counter
    k = 0

    # Loop through the larger image dimensions in steps of patch_size
    for i in range(0, 2048, patch_size):
        for j in range(0, 2048, patch_size):
            # If the number of patches exceeds the total number of provided images, exit
            # print(i,j)
            # print(k)
            if k >= len(imgs_of_a_class) or k == exit_patch:
                break

            # Place the patch into the compressed_image
            compressed_image[i:i + patch_size, j:j + patch_size] = imgs_of_a_class[k][:patch_size, :patch_size]

            # # Increment the patch counter
            k += 1

    return compressed_image



def func(inp_imgs:list):

    print(inp_imgs)
    output_images = []
    new_video = []
    for filename in inp_imgs:
        
        masked_images, images = read_and_store(path=filename) 
        
        patch_size = 128

        masked_images_array = np.stack(masked_images)
        new_video.extend(masked_images)

        compressed_image = compress_image(masked_images_array, patch_size, len(masked_images_array))

        output_images.append(compressed_image)
    # result.release()
    # print(f"Length of new videos: {len(new_video)}")
    save_video_from_frames(new_video, "Video.mp4")
    masked_images_np = np.stack(output_images)
    # print(masked_images_np.shape)
    masked_images_tensor = torch.from_numpy(masked_images_np).to(torch.float32).transpose(1, -1).to(device)
    print(masked_images_tensor.shape)
    with torch.no_grad():
        out = model(masked_images_tensor)

    idxs = torch.argmax(out, dim=-1)
    # print(idxs)
    # print("Result: \n")
    result = ""
    for idx in idxs:
        # print(cls_dict[str(idx.item())], end=" ")
        result+=cls_dict[str(idx.item())] +  " "
    print(result)

    response = chat(result)

    return "Video.mp4", response

interface = gr.Interface(fn=func, inputs=gr.File(file_count='multiple'), outputs=["playable_video", 'text'])
if __name__ == "__main__":
    interface.launch()