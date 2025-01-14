# Face Mask Detector

This project is a simple face mask detector built using deep learning with TensorFlow/Keras and OpenCV. It detects whether a person in a given image or video is wearing a mask or not.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The Face Mask Detector uses a pre-trained MobileNetV2 model to classify faces as either 'with mask' or 'without mask'. It uses a deep learning approach to identify face regions in images or video streams and applies a binary classification for mask presence.

### Features:

- Real-time face mask detection using a webcam or video feed.
- Ability to process images and classify masks.
- Customizable for various datasets and additional preprocessing steps.

## Installation

To install the necessary dependencies, you can use pip and a `requirements.txt` file.

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/face_mask_detector.git
   cd face_mask_detector

   ```

2. Create a virtual environment (optional):

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`

   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. To train the model, create a dataset folder in the root directory containing with_mask and without_mask subfolders.
