# 🧠 Brain Tumor Detection — Desktop Application

A Python-based desktop application for **binary classification of brain MRI images** using a customized **EfficientNet-B0** deep-learning model.

The application provides a simple file-selection workflow where a user selects an MRI image and the model predicts whether the image belongs to one of two classes:

```text
no_tumor
tumor
```

The application also reports model inference time and automatically uses a CUDA-enabled GPU when one is available.

> **Medical disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not be used to make clinical decisions.

## ✨ Features

- 🧠 Brain MRI image classification
- 🤖 EfficientNet-B0 transfer-learning architecture
- 🔎 Binary classification:
  - `no_tumor`
  - `tumor`
- 🖥️ Desktop file-selection interface using Tkinter
- 🖼️ Automatic MRI image preprocessing
- ⚡ CUDA/GPU support when available
- 🧮 CPU fallback when CUDA is unavailable
- ⏱️ Inference-time measurement
- 💾 Optional loading of a trained PyTorch checkpoint
- 🖥️ Console-based prediction output

## 🏗️ Architecture

```text
                 ┌─────────────────────┐
                 │     User / GUI      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Select MRI Image  │
                 │       Tkinter       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Image Preprocessing │
                 │                     │
                 │ Resize 224 × 224    │
                 │ RGB Conversion      │
                 │ Tensor Conversion   │
                 │ ImageNet Normalize  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    EfficientNet-B0  │
                 │                     │
                 │ Pretrained Backbone │
                 │ Custom 2-Class FC   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Classification   │
                 │                     │
                 │     no_tumor        │
                 │        or           │
                 │       tumor         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Prediction + Time   │
                 │      Console        │
                 └─────────────────────┘
```

## 🧠 Model

The application uses **EfficientNet-B0** from the `efficientnet-pytorch` package.

The pretrained EfficientNet-B0 classifier is modified by replacing its final fully connected layer:

```python
num_features = model._fc.in_features
model._fc = torch.nn.Linear(
    num_features,
    len(class_names)
)
```

Since:

```python
class_names = ['no_tumor', 'tumor']
```

the final layer produces two class outputs.

### Transfer Learning

The project uses the ImageNet-pretrained EfficientNet-B0 architecture as the starting point and replaces the original classification head for the two target classes.

A custom trained checkpoint can optionally be loaded:

```python
model = load_model(
    weights_path='brain_tumor_model.pth'
)
```

## 🔄 Prediction Pipeline

The application follows this workflow:

```text
MRI Image
   │
   ▼
File Selection
   │
   ▼
RGB Conversion
   │
   ▼
Resize → 224 × 224
   │
   ▼
Convert to Tensor
   │
   ▼
ImageNet Normalization
   │
   ▼
Add Batch Dimension
   │
   ▼
EfficientNet-B0
   │
   ▼
Argmax Classification
   │
   ├───────────────┐
   ▼               ▼
no_tumor          tumor
```

## 🖼️ Image Preprocessing

Images are converted to RGB and transformed using:

```python
transforms.Resize((224, 224))
transforms.ToTensor()
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

A batch dimension is then added before inference:

```python
image.unsqueeze(0)
```

This produces the expected input shape for the EfficientNet model.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application and inference logic |
| PyTorch | Deep-learning framework |
| EfficientNet-PyTorch | EfficientNet-B0 implementation |
| Torchvision | Image preprocessing |
| Tkinter | Desktop file-selection interface |
| Pillow | MRI image loading and RGB conversion |
| CUDA | Optional GPU acceleration |

## 📂 Project Structure

```text
Brain-Tumor-Detection-Desktop-Application/
│
├── braintumordetection.py
└── input.jpg
```

### `braintumordetection.py`

The main application containing:

- Model initialization
- EfficientNet-B0 customization
- Image selection
- Image preprocessing
- Model inference
- Device selection
- Prediction
- Inference-time measurement

### `input.jpg`

A sample input image included with the project.

## ⚙️ Requirements

The supplied project does not currently contain a `requirements.txt`.

Install the required Python packages with:

```bash
pip install torch torchvision efficientnet-pytorch pillow
```

Tkinter is included with most standard Python installations on Windows. On some Linux distributions, it may need to be installed separately through the operating system's package manager.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Brain-Tumor-Detection-Desktop-Application
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision efficientnet-pytorch pillow
```

For GPU acceleration, install the PyTorch build appropriate for your CUDA version from the official PyTorch installation instructions.

## ▶️ Run the Application

Run:

```bash
python braintumordetection.py
```

A file-selection dialog will appear.

Select an MRI image to run classification.

The application prints output similar to:

```text
Uploaded file: /path/to/mri_image.jpg
Prediction: tumor
Inference Time: 0.1234 seconds
```

If CUDA is available, it also reports maximum GPU memory allocated during the process.

## ⚡ CPU vs GPU

The application automatically selects the available computation device:

```python
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
```

Therefore:

```text
CUDA available
      ↓
Use GPU

CUDA unavailable
      ↓
Use CPU
```

No manual device-selection flag is required.

## 💾 Using a Trained Model Checkpoint

The current code initializes EfficientNet-B0 using pretrained ImageNet weights.

For a custom brain-tumor classifier checkpoint, provide its path to:

```python
load_model()
```

For example:

```python
model = load_model(
    weights_path='brain_tumor_model.pth'
)
```

The checkpoint must be compatible with the modified EfficientNet-B0 architecture containing the two-class output layer.

## 📊 Prediction Output

The application currently returns:

### Prediction

One of:

```text
no_tumor
```

or:

```text
tumor
```

### Inference Time

The elapsed time for the forward pass is measured using:

```python
start_time = time.time()

outputs = model(input_tensor)

end_time = time.time()
```

and displayed in seconds.

### GPU Memory

When CUDA is available, the application also reports:

```text
Max GPU memory allocated
```

## 🔬 Technical Details

### Model Input

```text
224 × 224 RGB image
```

### Output Classes

```text
2
```

### Classification Method

The predicted class is selected using the maximum model output:

```python
_, predicted = torch.max(outputs, 1)
```

The resulting class index is mapped to:

```python
class_names = ['no_tumor', 'tumor']
```

## ⚠️ Limitations

This is a relatively simple inference application and has several limitations:

- The repository does not contain the training pipeline or training dataset.
- The repository does not contain a trained custom checkpoint.
- The default EfficientNet-B0 initialization uses ImageNet-pretrained weights.
- The application does not display prediction probabilities or confidence scores.
- The application performs binary classification only.
- There is no model-performance evaluation interface.
- There is no confusion matrix, ROC curve, or classification report.
- The application does not perform medical image segmentation.
- The application does not localize a tumor within an MRI.
- Input validation and error handling can be expanded.
- The application currently prints results to the console rather than displaying them in a full prediction GUI.

## 🔮 Future Improvements

Potential enhancements include:

- Add a complete training pipeline for the brain MRI dataset.
- Add a trained model checkpoint to the deployment package.
- Display prediction probabilities/confidence scores.
- Build a complete Tkinter prediction dashboard.
- Add image preview before classification.
- Add Grad-CAM or similar explainability visualization.
- Highlight image regions influencing the prediction.
- Add batch MRI classification.
- Export prediction results to CSV or PDF.
- Add model evaluation metrics.
- Add validation for supported image formats.
- Add robust exception handling.
- Package the application as a standalone executable with PyInstaller.
- Add GPU/CPU configuration controls.
- Add automated tests.
- Add a `requirements.txt` for reproducible environments.

## 🩺 Medical Disclaimer

This project is an **educational/research demonstration of image classification using deep learning**.

It is **not a medical device, diagnostic system, or substitute for professional medical advice**.

Predictions generated by the application should not be used to diagnose, rule out, or treat brain tumors or any other medical condition. Clinical decisions must be made by qualified healthcare professionals using appropriate medical evaluation and validated diagnostic systems.

## 🔐 Privacy

The application uses a local file-selection dialog to select an image and performs model inference locally.

The project does not implement a remote image-upload service or cloud-based image storage.

Users should still avoid processing identifiable or sensitive medical images in environments where they do not have appropriate authorization.

## 📜 License

If your repository includes an MIT License, use:

```markdown
This project is licensed under the [MIT License](LICENSE).

You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this software, subject to the conditions of the MIT License.
```

---

**Built with Python · PyTorch · EfficientNet-B0 · Torchvision · Tkinter · Pillow**
