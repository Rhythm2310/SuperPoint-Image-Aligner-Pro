# 🌟 SuperPoint Image Aligner Pro 🌟

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)  
**SuperPoint Image Aligner Pro** is a cutting-edge image alignment tool designed for professionals and researchers. Built on top of the **SuperPoint** model from Hugging Face's `transformers` library, this tool provides advanced features such as:

- **Scenario Presets**: Predefined parameter configurations for different use cases.
- **Multi-Pass Alignment**: Iterative refinement for improved accuracy.
- **Detailed Parameter Documentation**: Comprehensive tooltips and descriptions for fine-tuning.
- **Advanced Tuning Controls**: Customize keypoint detection, feature matching, and geometric validation parameters.
- **Auto AI**: Automatic parameter optimization based on image characteristics.

---

## 📚 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Advanced Tuning Parameters](#advanced-tuning-parameters)
- [In-Depth Calculation Explanations](#in-depth-calculation-explanations)
- [Code Structure](#code-structure)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

1. **SuperPoint Feature Detection**:
   - Extracts keypoints and descriptors using the **SuperPoint** model.
   - Filters keypoints based on confidence scores and maximum keypoint limits.

2. **Auto AI Parameter Optimization**:
   - Analyzes image characteristics such as texture, brightness, and contrast.
   - Dynamically adjusts alignment parameters for optimal results.

3. **Multi-Pass Alignment Pipeline**:
   - Supports up to 5 alignment passes for iterative refinement.
   - Uses FLANN-based feature matching and RANSAC for robust homography estimation.

4. **Professional GUI**:
   - Built with **PyQt6** for an intuitive user interface.
   - Includes input configuration, advanced tuning parameters, and real-time logging.

5. **Scenario Presets**:
   - Predefined settings for various scenarios:
     - General Purpose
     - Low-texture Images
     - Fast Processing
     - High Precision
     - Satellite Imagery
     - Medical Images

---

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster inference)

### Dependencies

Install the required dependencies using `pip`:

```bash
pip install torch transformers opencv-python pillow PyQt6
```

### Clone the Repository

```bash
git clone https://github.com/Rhythm2310/SuperPoint-Image-Aligner-Pro.git
cd SuperPoint-Image-Aligner-Pro
```

---

## 🚀 Usage

### Running the GUI

To launch the graphical user interface:

```bash
python main.py
```

### Input Configuration

1. **Reference Image**: Select the reference image to which other images will be aligned.
2. **Target Images**: Select one or more target images for alignment.
3. **Output Directory**: Choose the directory where aligned images will be saved.

### Advanced Tuning Parameters

The GUI provides sliders and input fields for adjusting key parameters, including:

| Parameter               | Description                                                                                   | Default Value |
|-------------------------|-----------------------------------------------------------------------------------------------|---------------|
| `min_matches`           | Minimum number of geometrically consistent matches required for alignment.                    | 15            |
| `ratio_threshold`       | Lowe's ratio test threshold for feature matching.                                             | 0.7           |
| `flann_trees`           | Number of FLANN index trees (KD-Tree).                                                        | 5             |
| `flann_checks`          | Number of FLANN search checks.                                                                | 150           |
| `usac_thresh`           | USAC reprojection error threshold (pixels).                                                   | 3.0           |
| `max_keypoints`         | Maximum number of keypoints retained per image.                                               | 1200          |
| `match_distance_thresh` | Absolute descriptor distance threshold.                                                       | 0.65          |
| `min_inlier_ratio`      | Minimum ratio of inlier matches to total matches.                                             | 0.25          |
| `num_passes`            | Number of alignment iterations (1-5).                                                         | 1             |

### Auto AI

Enable the "Auto AI" checkbox to automatically analyze each image and optimize alignment parameters based on its characteristics.

### Logging

The processing log displays real-time updates, including:
- Detected keypoints and matches.
- Homography estimation results.
- Saved output paths.

---

## 🔍 In-Depth Calculation Explanations

### Feature Extraction

The **SuperPoint** model extracts keypoints and descriptors from images. The keypoints are filtered based on their confidence scores. If the number of keypoints exceeds the specified maximum (`max_keypoints`), only the top-scoring keypoints are retained.

```
Filtered Keypoints = Keypoints[argsort(Scores)[-max_keypoints:]]
```

### Auto AI: Image Analysis

The `analyze_image` function evaluates image characteristics such as texture, brightness, and contrast to suggest optimal parameters. Texture sharpness is measured using the variance of the Laplacian:

```
Laplacian Variance = Var(Laplacian(Image))
```

Brightness and contrast are calculated as:

```
Mean Intensity = Sum(Pixel Values) / Total Pixels
Contrast = Sqrt(Sum((Pixel Values - Mean Intensity)^2) / Total Pixels)
```

Based on these metrics, the function adjusts parameters like `min_matches`, `ratio_threshold`, and `flann_checks`.

### Feature Matching

The FLANN-based matcher uses the k-nearest neighbors algorithm to find the best matches between descriptors. Lowe's ratio test ensures that only high-quality matches are retained:

```
m.distance < ratio_threshold * n.distance
```

Additionally, an absolute distance threshold is applied:

```
m.distance < match_distance_thresh
```

### Homography Estimation

The homography matrix `H` is estimated using RANSAC with the USAC algorithm. The reprojection error threshold determines the maximum allowable error for inliers:

```
Reprojection Error = ||x' - H * x||
```

where `x` and `x'` are corresponding points in the source and destination images.

The inlier ratio is calculated as:

```
Inlier Ratio = Number of Inliers / Total Matches
```

Only alignments with an inlier ratio greater than `min_inlier_ratio` are accepted.

---

## 🧩 Code Structure

- **Model Initialization**: Loads the SuperPoint model and sets up the device (CPU/GPU).
- **Feature Extraction**: Extracts and filters keypoints and descriptors using SuperPoint.
- **Alignment Logic**: Implements multi-pass alignment with FLANN-based matching and RANSAC.
- **Worker Thread**: Handles alignment in a separate thread to prevent GUI freezing.
- **GUI**: Built with PyQt6 for an interactive and user-friendly experience.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## 📜 License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

---

### 🎉 Acknowledgments

- Special thanks to the **Hugging Face Transformers** team for providing the **SuperPoint** model.
- Thanks to the **OpenCV** community for their robust computer vision tools.

---

### 📝 Citations

If you use this tool in your research, please cite it as follows:

```bibtex
@software{SuperPointImageAlignerPro,
  author = {Abdullah Al Siam},
  title = {SuperPoint Image Aligner Pro},
  year = {2025},
  url = {https://github.com/Rhythm2310/SuperPoint-Image-Aligner-Pro},
  version = {1.0.0}
}
```
