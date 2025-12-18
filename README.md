# Face Blur YOLO 🎭

A clean and simple Python project for **blurring faces in videos** using  
**YOLOv8 face detection** and **OpenCV**.

The project automatically selects **CPU or GPU (CUDA)** and supports **any YOLOv8 face `.pt` model**.

---

## ✨ Features

- 🎯 Accurate face detection with YOLOv8
- ⚡ Automatic CPU / GPU selection
- 🔁 Process videos frame-by-frame
- 🔧 Easily switch models without code changes
- 🧩 Clean, modular, GitHub-ready structure
- 🖥️ Works on Windows / Linux / macOS

---

## 📁 Project Structure

```
face-blur/
├── face_blur/
│   ├── __init__.py
│   └── processor.py
│
├── examples/
│   └── blur_video.py
│
├── models/
│   └── yolov8s-face-lindevs.pt #added manually
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Requirements

- Python 3.8 – 3.11
- OpenCV
- PyTorch
- Ultralytics YOLOv8

---

## 📦 Installation

```bash
git clone https://github.com/yourname/face-blur.git
cd face-blur
python -m venv venv
```

### Activate virtual environment

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Face Detection Models

Any YOLOv8 face detection `.pt` model is supported.

Default model:
```
models/yolov8s-face-lindevs.pt
```

Other examples:
- yolov8n-face.pt
- yolov8s-face.pt
- yolov8m-face.pt
- Custom trained models

Model filename does not need to match the code.

---

## 🚀 Usage

```bash
python examples/blur_video.py --input input.mp4 --output output.mp4
```

### Use a different model

```bash
python examples/blur_video.py --model models/yolov8n-face.pt
```

### Show all options

```bash
python examples/blur_video.py --help
```

| Argument | Description | Default |
|--------|------------|--------|
| `--input` | Input video path | `input.mp4` |
| `--output` | Output video path | `output_blur.mp4` |
| `--model` | Path to YOLOv8 face model | `models/yolov8s-face-lindevs.pt` |
| `--analyze-width` | Resize width | `640` |
| `--detect-every` | Detect every N frames | `1` |
| `--conf` | Confidence threshold | `0.05` |

---

## 🖥️ Device Selection

The device is selected automatically:

- CUDA available → GPU
- CUDA not available → CPU

---

## 🧪 Notes
- YOLOv8 face models are NOT included in this repository and must be added manually.
- `__pycache__/` folders are created automatically and should not be committed
- Input video can be relative or absolute path
- Output video keeps original resolution and FPS

---

## 📄 License

MIT License

---

## ⭐ Credits

- Ultralytics YOLOv8
- OpenCV
- PyTorch
