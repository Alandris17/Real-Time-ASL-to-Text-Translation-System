# American Sign Language to Text (ASLTT)

## Project Description

This project translates American Sign Language (ASL) gestures captured via a camera into text.

The workflow consists of:

1. **Data Collection** – Capture gesture images.
2. **Preprocessing** – Augment images and extract hand landmarks.
3. **Model Training** – Train a gesture recognition model.
4. **Real-time Prediction** – Use the trained model to predict gestures live.

---

## Dataset Setup (Important)

Due to GitHub size limitations, the `Data/` and `Data_preprocessed/` folders are **not included** in this repository.

### Step 1: Download the Dataset

Download the dataset from Google Drive:

**[https://drive.google.com/drive/folders/1IIwpMUTbbebH_tyIpP_asu07n-6l1agG?usp=share_link]**

### Step 2: Extract the Dataset

After downloading:

1. Extract the archive.
2. Place the `Data/` folder in the root of the project.

Your project structure should look like:

```
project_root/
│── Data/
│── data_collection.py
│── flipper.py
│── model_training.py
│── hand_gestures.py
```

---

## Installation and Setup

### 1. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 2. Activate the Virtual Environment

* macOS/Linux:

```bash
source .venv/bin/activate
```

* Windows:

```bash
.\.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Use the Project

Run scripts **in this exact order**

---

### 1. Data Collection (Optional) – `data_collection.py`

Use this if you want to add new gestures.

* Modify:

```python
folder = "Data/YourGesture"
```

* Run:

```bash
python data_collection.py
```

* Controls:

  * Press **S** → Save image
  * Press **Q** → Quit

---

### 2. Preprocessing – `flipper.py`

This step generates `Data_preprocessed/` from `Data/`.

* Ensure:

```python
dataset_path = "Data"
```

* Run:

```bash
python flipper.py
```

Output:

```
Data_preprocessed/
```

---

### 3. Model Training – `model_training.py`

This step:

* Extracts hand landmarks

* Trains the model

* Saves outputs

* Ensure:

```python
dataset_path = "Data_preprocessed"
```

* Run:

```bash
python model_training.py
```

Outputs:

```
landmark_gestures/landmarks_cache.pkl
kt_dir/
exported_model/gesture_model.keras
```

---

### 4. Real-time Prediction – `hand_gestures.py`

Uses webcam + trained model.

* Ensure:

```python
dataset_path = "Data_preprocessed"
model_path = "exported_model/gesture_model.keras"
```

* For laptop usage:

```python
if __name__ == "__main__":
    vision_laptop()
```

* Run:

```bash
python hand_gestures.py
```

* Press **Q** to quit

---

## Project Structure

```
project_root/
│── Data/                  # Downloaded from Google Drive
│── Data_preprocessed/     # Generated locally
│── exported_model/
│── landmark_gestures/
│── kt_dir/
│── data_collection.py
│── flipper.py
│── model_training.py
│── hand_gestures.py
│── requirements.txt
```

---

## Technologies Used

* OpenCV
* Mediapipe
* NumPy
* TensorFlow / Keras
* Keras-Tuner
* Scikit-Learn

---

## Notes

* Do **not upload** `Data/` or `Data_preprocessed/` to GitHub
* Always regenerate preprocessing locally
* Model must be retrained if dataset changes

---