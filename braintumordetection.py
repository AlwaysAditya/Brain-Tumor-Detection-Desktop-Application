import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import time

# --- step 1: Setup class names for your model
# Ensure these match the folder names/classes you used during training
class_names = ['no_tumor', 'tumor']

# --- Step 2: Function to select an image using file dialog
def upload_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select MRI Image")
    if not file_path:
        raise Exception("No file selected.")
    return file_path

# --- Step 3: Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# --- Step 4: EfficientNet model initialization and customization
def load_model(weights_path=None):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_features = model._fc.in_features
    model._fc = torch.nn.Linear(num_features, len(class_names))  # Output for two classes
    if weights_path:
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state)
    model.eval()
    return model

# --- Step 5: Single image prediction function
def predict_image(model, device, image_path):
    input_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_tensor)
        end_time = time.time()
        _, predicted = torch.max(outputs, 1)
        pred_class = class_names[predicted.item()]
        inference_time = end_time - start_time
    return pred_class, inference_time

# --- Step 6: Main execution
def main():
    # DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model (set weights path if you have a trained checkpoint)
    # Example: model = load_model(weights_path='brain_tumor_model.pth')
    model = load_model()
    model.to(device)
    
    # Upload and predict
    image_path = upload_image()
    print(f"Uploaded file: {image_path}")
    pred_class, inference_time = predict_image(model, device, image_path)
    print(f"Prediction: {pred_class}")
    print(f"Inference Time: {inference_time:.4f} seconds")
    
    if torch.cuda.is_available():
        print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated(device) / 1024 ** 2 :.2f} MB")

if __name__ == '__main__':
    main()
