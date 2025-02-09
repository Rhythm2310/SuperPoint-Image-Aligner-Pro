#!/usr/bin/env python3
"""
SuperPoint Image Aligner Pro
Professional image alignment tool with:
- Scenario presets
- Multi-pass alignment
- Detailed parameter documentation
- Advanced tuning controls
- Auto AI: Automatic parameter optimization
"""
import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                             QLineEdit, QFileDialog, QVBoxLayout, QHBoxLayout,
                             QLabel, QSpinBox, QDoubleSpinBox, QTextEdit,
                             QGroupBox, QGridLayout, QCheckBox, QButtonGroup,
                             QRadioButton, QStatusBar)
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor


# Model Initialization
print("Initializing SuperPoint model...")
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using {device} for inference")


# Feature Extraction
def extract_superpoint_features(cv2_image):
    """Extract and filter SuperPoint features with adaptive thresholding"""
    pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    inputs = processor(pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    processed = processor.post_process_keypoint_detection(
        outputs, [(cv2_image.shape[0], cv2_image.shape[1])]
    )
    return (
        processed[0]["keypoints"].cpu().numpy(),
        processed[0]["scores"].cpu().numpy(),
        processed[0]["descriptors"].cpu().numpy()
    )


# Auto AI: Image Analysis and Parameter Tuning
def analyze_image(image):
    """
    Analyze image characteristics and suggest optimal parameters
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # Texture/Sharpness
    mean_intensity = np.mean(gray)  # Brightness
    std_intensity = np.std(gray)  # Contrast

    params = {
        'min_matches': 15,
        'ratio_threshold': 0.7,
        'flann_trees': 5,
        'flann_checks': 150,
        'usac_thresh': 3.0,
        'max_keypoints': 1200,
        'match_distance_thresh': 0.65,
        'min_inlier_ratio': 0.25,
        'num_passes': 1
    }

    # Adjust parameters based on image characteristics
    if laplacian_var < 50:  # Low texture
        params['min_matches'] = 8
        params['ratio_threshold'] = 0.75
        params['flann_trees'] = 8
        params['flann_checks'] = 300
        params['max_keypoints'] = 2000
        params['match_distance_thresh'] = 0.75
        params['min_inlier_ratio'] = 0.2
        params['num_passes'] = 2
    elif laplacian_var > 200:  # High texture
        params['min_matches'] = 20
        params['ratio_threshold'] = 0.6
        params['flann_trees'] = 8
        params['flann_checks'] = 500
        params['max_keypoints'] = 2000
        params['match_distance_thresh'] = 0.5
        params['min_inlier_ratio'] = 0.4
        params['num_passes'] = 3

    if std_intensity < 20:  # Low contrast
        params['usac_thresh'] = 5.0
        params['min_inlier_ratio'] = max(params['min_inlier_ratio'], 0.15)

    return params


# Enhanced Alignment Logic
def align_images(ref_image_path, image_paths, output_dir, min_matches=10,
                 ratio_threshold=0.7, usac_thresh=5.0, flann_trees=5,
                 flann_checks=100, max_keypoints=1000, match_distance_thresh=0.7,
                 min_inlier_ratio=0.3, num_passes=1, use_auto_ai=False, logger=None):
    """
    Advanced multi-pass image alignment pipeline
    Parameters:
    num_passes (int): Number of alignment iterations (1-5)
    [1: Single pass, 2-5: Iterative refinement]
    """
    # Load reference image
    ref_img = cv2.imread(ref_image_path)
    if ref_img is None:
        logger(f"Error loading reference image: {ref_image_path}")
        return
    # Initial feature extraction
    ref_kp, ref_scores, ref_desc = extract_superpoint_features(ref_img)
    if len(ref_kp) > max_keypoints:
        idx = np.argsort(ref_scores)[-max_keypoints:]
        ref_kp = ref_kp[idx]
        ref_desc = ref_desc[idx]
    logger(f"Initial Reference: {len(ref_kp)} keypoints")
    # FLANN configuration
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=flann_trees)
    search_params = dict(checks=flann_checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    for pass_num in range(1, num_passes + 1):
        logger(f"\n=== Starting Alignment Pass {pass_num}/{num_passes} ===")
        current_ref = {
            'kp': ref_kp,
            'desc': ref_desc.astype(np.float32),
            'img': ref_img
        }
        for idx, img_path in enumerate(image_paths):
            logger(f"\nProcessing image {idx + 1}/{len(image_paths)}")
            curr_img = cv2.imread(img_path)
            if curr_img is None:
                logger(f"Error loading image: {img_path}")
                continue
            # Auto AI: Analyze and adjust parameters
            if use_auto_ai:
                auto_params = analyze_image(curr_img)
                min_matches = auto_params['min_matches']
                ratio_threshold = auto_params['ratio_threshold']
                flann_trees = auto_params['flann_trees']
                flann_checks = auto_params['flann_checks']
                usac_thresh = auto_params['usac_thresh']
                max_keypoints = auto_params['max_keypoints']
                match_distance_thresh = auto_params['match_distance_thresh']
                min_inlier_ratio = auto_params['min_inlier_ratio']
                num_passes = auto_params['num_passes']
                logger(f"Auto AI Adjusted Parameters: {auto_params}")

            # Feature extraction
            curr_kp, curr_scores, curr_desc = extract_superpoint_features(curr_img)
            if len(curr_kp) > max_keypoints:
                idx_filter = np.argsort(curr_scores)[-max_keypoints:]
                curr_kp = curr_kp[idx_filter]
                curr_desc = curr_desc[idx_filter]
            curr_desc = curr_desc.astype(np.float32)
            logger(f"Current: {len(curr_kp)} keypoints")
            # Feature matching
            matches = flann.knnMatch(current_ref['desc'], curr_desc, k=2)
            good_matches = []
            for m, n in matches:
                if (m.distance < ratio_threshold * n.distance and
                        m.distance < match_distance_thresh):
                    good_matches.append(m)
            logger(f"Matches: {len(good_matches)} (min {min_matches})")
            if len(good_matches) < min_matches:
                logger("Insufficient matches - skipping")
                continue
            # Homography estimation
            src_pts = np.float32([current_ref['kp'][m.queryIdx] for m in good_matches])
            dst_pts = np.float32([curr_kp[m.trainIdx] for m in good_matches])
            H, mask = cv2.findHomography(
                dst_pts, src_pts,
                method=cv2.USAC_DEFAULT,
                ransacReprojThreshold=usac_thresh,
                maxIters=10000,
                confidence=0.999
            )
            if H is None:
                logger("Homography estimation failed")
                continue
            # Inlier validation
            inlier_ratio = np.sum(mask) / len(mask)
            logger(f"Inlier ratio: {inlier_ratio:.2f} (min {min_inlier_ratio})")
            if inlier_ratio < min_inlier_ratio:
                logger("Low inlier ratio - rejecting alignment")
                continue
            # Image warping
            h, w = current_ref['img'].shape[:2]
            aligned = cv2.warpPerspective(curr_img, H, (w, h))
            output_path = os.path.join(output_dir, f"pass{pass_num}_aligned_{idx + 1}.jpg")
            cv2.imwrite(output_path, aligned)
            logger(f"Saved: {output_path}")
            # Update reference for next pass
            if pass_num < num_passes:
                ref_img = aligned
                ref_kp = curr_kp
                ref_desc = curr_desc


# Worker Thread
class AlignmentWorker(QObject):
    finished = pyqtSignal()
    log = pyqtSignal(str)

    def __init__(self, ref_image_path, image_paths, output_dir, **params):
        super().__init__()
        self.params = {
            'ref_image_path': ref_image_path,
            'image_paths': image_paths,
            'output_dir': output_dir,
            **params
        }

    def run(self):
        def logger(message):
            self.log.emit(message)

        try:
            align_images(**self.params, logger=logger)
        except Exception as e:
            self.log.emit(f"Critical Error: {str(e)}")
        self.finished.emit()


# Professional GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SuperPoint Aligner Pro")
        self.resize(1400, 1000)
        self.init_ui()
        self.worker_thread = None

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(20)

        # Preset Section
        preset_group = QGroupBox("Scenario Presets")
        preset_layout = QHBoxLayout()
        self.presets = {
            "General Purpose": {
                'min_matches': 15,
                'ratio_threshold': 0.7,
                'flann_trees': 5,
                'flann_checks': 150,
                'usac_thresh': 3.0,
                'max_keypoints': 1200,
                'match_distance_thresh': 0.65,
                'min_inlier_ratio': 0.25,
                'num_passes': 1
            },
            "Low-texture Images": {
                'min_matches': 8,
                'ratio_threshold': 0.75,
                'flann_trees': 8,
                'flann_checks': 300,
                'usac_thresh': 5.0,
                'max_keypoints': 2000,
                'match_distance_thresh': 0.75,
                'min_inlier_ratio': 0.2,
                'num_passes': 2
            },
            "Fast Processing": {
                'min_matches': 10,
                'ratio_threshold': 0.8,
                'flann_trees': 3,
                'flann_checks': 50,
                'usac_thresh': 5.0,
                'max_keypoints': 500,
                'match_distance_thresh': 0.8,
                'min_inlier_ratio': 0.2,
                'num_passes': 1
            },
            "High Precision": {
                'min_matches': 20,
                'ratio_threshold': 0.6,
                'flann_trees': 8,
                'flann_checks': 500,
                'usac_thresh': 1.0,
                'max_keypoints': 2000,
                'match_distance_thresh': 0.5,
                'min_inlier_ratio': 0.4,
                'num_passes': 3
            },
            "Satellite Imagery": {
                'min_matches': 25,
                'ratio_threshold': 0.65,
                'flann_trees': 6,
                'flann_checks': 200,
                'usac_thresh': 2.0,
                'max_keypoints': 1500,
                'match_distance_thresh': 0.6,
                'min_inlier_ratio': 0.3,
                'num_passes': 2
            },
            "Medical Images": {
                'min_matches': 12,
                'ratio_threshold': 0.7,
                'flann_trees': 7,
                'flann_checks': 400,
                'usac_thresh': 1.5,
                'max_keypoints': 1800,
                'match_distance_thresh': 0.55,
                'min_inlier_ratio': 0.35,
                'num_passes': 1
            }
        }
        self.preset_btns = {}
        for preset_name in self.presets:
            btn = QPushButton(preset_name)
            btn.clicked.connect(lambda _, name=preset_name: self.apply_preset(name))
            preset_layout.addWidget(btn)
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # Input Section
        input_group = QGroupBox("Input Configuration")
        input_layout = QGridLayout()

        # Reference Image
        self.ref_line = QLineEdit()
        self.ref_line.setPlaceholderText("Select reference image...")
        ref_btn = QPushButton("Browse...")
        ref_btn.clicked.connect(self.browse_ref)
        input_layout.addWidget(QLabel("Reference Image:"), 0, 0)
        input_layout.addWidget(self.ref_line, 0, 1)
        input_layout.addWidget(ref_btn, 0, 2)

        # Target Images
        self.target_line = QLineEdit()
        self.target_line.setPlaceholderText("Select target images...")
        target_btn = QPushButton("Browse...")
        target_btn.clicked.connect(self.browse_targets)
        input_layout.addWidget(QLabel("Target Images:"), 1, 0)
        input_layout.addWidget(self.target_line, 1, 1)
        input_layout.addWidget(target_btn, 1, 2)

        # Output Directory
        self.out_line = QLineEdit()
        self.out_line.setPlaceholderText("Select output directory...")
        out_btn = QPushButton("Browse...")
        out_btn.clicked.connect(self.browse_output)
        input_layout.addWidget(QLabel("Output Directory:"), 2, 0)
        input_layout.addWidget(self.out_line, 2, 1)
        input_layout.addWidget(out_btn, 2, 2)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Tuning Parameters
        tune_group = QGroupBox("Advanced Tuning Parameters")
        tune_layout = QGridLayout()

        # Core Parameters
        self.min_matches = self.create_spin(1, 1000, 10,
                                            "Minimum number of geometrically consistent matches required for alignment\n"
                                            "Higher values increase reliability but may reject valid alignments\n"
                                            "Typical values: 10-50\n"
                                            "Low-texture: 5-15, High-detail: 20-50")
        self.ratio_thresh = self.create_double_spin(0.1, 1.0, 0.7, 0.05,
                                                    "Lowe's ratio test threshold for feature matching\n"
                                                    "Lower values reduce false matches but decrease total matches\n"
                                                    "Recommended: 0.6-0.8\n"
                                                    "Noisy images: 0.75-0.85, Clean images: 0.6-0.7")

        # FLANN Parameters
        self.flann_trees = self.create_spin(1, 20, 5,
                                            "Number of FLANN index trees (KD-Tree)\n"
                                            "More trees improve search accuracy at memory cost\n"
                                            "Recommended: 4-8\n"
                                            "High-dimensional descriptors: 6-10, Low-memory: 3-5")
        self.flann_checks = self.create_spin(10, 1000, 100,
                                             "Number of FLANN search checks\n"
                                             "Higher values improve match quality at speed cost\n"
                                             "Balanced: 100-200, Precision: 300-500, Speed: 50-100")

        # Geometric Validation
        self.usac_thresh = self.create_double_spin(0.5, 20.0, 5.0, 0.5,
                                                   "USAC reprojection error threshold (pixels)\n"
                                                   "Maximum allowed pixel error for inlier classification\n"
                                                   "Typical: 1.0-10.0\n"
                                                   "Strict: 1-3, Lenient: 5-10, Low-res: 10-20")
        self.inlier_ratio = self.create_double_spin(0.1, 1.0, 0.3, 0.05,
                                                    "Minimum ratio of inlier matches to total matches\n"
                                                    "Filters poor geometric consensus\n"
                                                    "General: 0.2-0.4, Precision: 0.4-0.6\n"
                                                    "Challenging conditions: 0.15-0.25")

        # Feature Control
        self.max_keypoints = self.create_spin(100, 5000, 1000,
                                              "Maximum number of keypoints retained per image\n"
                                              "Balances feature richness vs computation time\n"
                                              "Typical: 800-1500\n"
                                              "Low-texture: 1500-3000, Real-time: 300-800")
        self.match_dist = self.create_double_spin(0.1, 1.0, 0.7, 0.05,
                                                  "Absolute descriptor distance threshold\n"
                                                  "Maximum allowed difference for valid matches\n"
                                                  "Combine with ratio test for robust matching\n"
                                                  "General: 0.6-0.8, Precision: 0.4-0.6\n"
                                                  "Low-quality images: 0.8-0.9")

        # Multi-pass Control
        self.num_passes = self.create_spin(1, 5, 1,
                                           "Number of alignment iterations\n"
                                           "1: Single pass\n"
                                           "2-3: Iterative refinement\n"
                                           "4-5: Complex multi-stage alignment\n"
                                           "Recommended: 1-3")

        # Layout parameters
        tune_layout.addWidget(QLabel("Core Parameters:"), 0, 0)
        tune_layout.addWidget(QLabel("Min Matches"), 1, 0)
        tune_layout.addWidget(self.min_matches, 1, 1)
        tune_layout.addWidget(QLabel("Ratio Threshold"), 2, 0)
        tune_layout.addWidget(self.ratio_thresh, 2, 1)
        tune_layout.addWidget(QLabel("FLANN Parameters:"), 0, 2)
        tune_layout.addWidget(QLabel("Index Trees"), 1, 2)
        tune_layout.addWidget(self.flann_trees, 1, 3)
        tune_layout.addWidget(QLabel("Search Checks"), 2, 2)
        tune_layout.addWidget(self.flann_checks, 2, 3)
        tune_layout.addWidget(QLabel("Geometric Validation:"), 3, 0)
        tune_layout.addWidget(QLabel("USAC Threshold"), 4, 0)
        tune_layout.addWidget(self.usac_thresh, 4, 1)
        tune_layout.addWidget(QLabel("Min Inlier Ratio"), 5, 0)
        tune_layout.addWidget(self.inlier_ratio, 5, 1)
        tune_layout.addWidget(QLabel("Feature Control:"), 3, 2)
        tune_layout.addWidget(QLabel("Max Keypoints"), 4, 2)
        tune_layout.addWidget(self.max_keypoints, 4, 3)
        tune_layout.addWidget(QLabel("Match Distance"), 5, 2)
        tune_layout.addWidget(self.match_dist, 5, 3)
        tune_layout.addWidget(QLabel("Multi-pass Control:"), 6, 0)
        tune_layout.addWidget(self.num_passes, 6, 1)

        # Auto AI Toggle
        self.auto_ai_checkbox = QCheckBox("Enable Auto AI")
        self.auto_ai_checkbox.setToolTip("Automatically analyze images and optimize parameters")
        tune_layout.addWidget(self.auto_ai_checkbox, 6, 2)

        tune_group.setLayout(tune_layout)
        layout.addWidget(tune_group)

        # Control Section
        self.run_btn = QPushButton("Start Alignment")
        self.run_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        self.run_btn.clicked.connect(self.start_alignment)
        layout.addWidget(self.run_btn)

        # Log Output
        log_group = QGroupBox("Processing Log")
        self.log_text = QTextEdit()
        self.log_text.setFont(QFont("Consolas", 10))
        log_group.setLayout(QVBoxLayout())
        log_group.layout().addWidget(self.log_text)
        layout.addWidget(log_group)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def create_spin(self, min_val, max_val, default, tooltip=""):
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setToolTip(tooltip)
        return spin

    def create_double_spin(self, min_val, max_val, default, step, tooltip=""):
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.setToolTip(tooltip)
        return spin

    def browse_ref(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.ref_line.setText(file_path)

    def browse_targets(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Target Images", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        if files:
            self.target_line.setText(";".join(files))

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if folder:
            self.out_line.setText(folder)

    def apply_preset(self, preset_name):
        params = self.presets[preset_name]
        self.min_matches.setValue(params['min_matches'])
        self.ratio_thresh.setValue(params['ratio_threshold'])
        self.flann_trees.setValue(params['flann_trees'])
        self.flann_checks.setValue(params['flann_checks'])
        self.usac_thresh.setValue(params['usac_thresh'])
        self.max_keypoints.setValue(params['max_keypoints'])
        self.match_dist.setValue(params['match_distance_thresh'])
        self.inlier_ratio.setValue(params['min_inlier_ratio'])
        self.num_passes.setValue(params['num_passes'])
        self.log_text.append(f"Applied preset: {preset_name}")

    def start_alignment(self):
        ref_path = self.ref_line.text().strip()
        targets = self.target_line.text().strip()
        output_dir = self.out_line.text().strip()
        if not all([ref_path, targets, output_dir]):
            self.log_text.append("Error: Missing required paths")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        params = {
            'min_matches': self.min_matches.value(),
            'ratio_threshold': self.ratio_thresh.value(),
            'flann_trees': self.flann_trees.value(),
            'flann_checks': self.flann_checks.value(),
            'usac_thresh': self.usac_thresh.value(),
            'max_keypoints': self.max_keypoints.value(),
            'match_distance_thresh': self.match_dist.value(),
            'min_inlier_ratio': self.inlier_ratio.value(),
            'num_passes': self.num_passes.value(),
            'use_auto_ai': self.auto_ai_checkbox.isChecked()
        }
        self.worker = AlignmentWorker(
            ref_image_path=ref_path,
            image_paths=[p.strip() for p in targets.split(';') if p.strip()],
            output_dir=output_dir,
            **params
        )
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self.update_log)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.run_btn.setEnabled(False)
        self.worker_thread.start()

    def update_log(self, message):
        self.log_text.append(message)
        if "Error" in message:
            self.log_text.setTextColor(QColor("red"))
        elif "Saved" in message:
            self.log_text.setTextColor(QColor("green"))
        else:
            self.log_text.setTextColor(QColor("black"))
        self.status_bar.showMessage(message)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
