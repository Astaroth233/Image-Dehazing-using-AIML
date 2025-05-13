# Filename: dehazing_gradio_app.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import os
from datetime import datetime

# Define the model architecture
class DehazeGenerator(nn.Module):
    def __init__(self):
        super(DehazeGenerator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.deconv1(x)))
        x = self.deconv2(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DehazeGenerator().to(device)
model.load_state_dict(torch.load("dehaze_finetuned_epoch_40.pth", map_location=device))
model.eval()

# Transform (no resizing here)
transform = transforms.ToTensor()

# Output directory
output_dir = "saved_outputs"
os.makedirs(output_dir, exist_ok=True)

# Max resolution for performance optimization
MAX_RESOLUTION = (720, 720)  # Moderate quality

def resize_if_needed(img):
    if img.size[0] > MAX_RESOLUTION[0] or img.size[1] > MAX_RESOLUTION[1]:
        img.thumbnail(MAX_RESOLUTION, Image.LANCZOS)
    return img

def dehaze_image(input_image):
    input_image = input_image.convert("RGB")
    original_size = input_image.size

    resized_image = resize_if_needed(input_image)
    image_tensor = transform(resized_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor).clamp(0, 1)

    output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
    output_image = output_image.resize(original_size)

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dehazed_{timestamp}.png"
    output_path = os.path.join(output_dir, filename)
    output_image.save(output_path)

    return output_image

# Gradio Interface
demo = gr.Interface(
    fn=dehaze_image,
    inputs=gr.Image(type="pil", label="Upload Hazy Image"),
    outputs=gr.Image(type="pil", label="Dehazed Image"),
    title="Image Dehazing using CNN",
    description="Upload a hazy image to see the dehazed output using a trained CNN model. Output will maintain original resolution, optimized for smooth performance."
)

if __name__ == "__main__":
    demo.launch()
