# FedGAN Usage Instructions

This document provides comprehensive instructions for using each component of the FedGAN project - a privacy-preserving federated learning framework for generating synthetic medical images using Generative Adversarial Networks.

## Table of Contents
1. [Setup and Installation](#setup-and-installation)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Evaluation and Analysis](#evaluation-and-analysis)
5. [Utilities](#utilities)
6. [Troubleshooting](#troubleshooting)

---

## Setup and Installation

### Prerequisites
- Python 3.11 or higher
- UV package manager (recommended) or pip
- TensorFlow 2.17
- GPU support (optional but recommended)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd FedGAN
   ```

2. **Install dependencies using UV (recommended):**
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install tensorflow==2.17 matplotlib pandas scikit-learn scipy tqdm
   ```

3. **Verify installation:**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

---

## Data Preparation

### 1. Data Preprocessing (`preprocessing.py`)

**Purpose:** Convert raw medical images to TFRecord format with enhancement preprocessing.

**Usage:**
```bash
python preprocessing.py <input_folder> <output_tfrecord> <resolution> <bin_size> [labels_csv]
```

**Parameters:**
- `input_folder`: Directory containing raw PNG/JPG images
- `output_tfrecord`: Output TFRecord file path (e.g., `data/processed.tfrecord`)
- `resolution`: Target image size (e.g., `128` for 128×128)
- `bin_size`: Pixel quantization level (e.g., `16`)
- `labels_csv`: (Optional) CSV file with image labels

**Example:**
```bash
# Without labels
python preprocessing.py data/raw_images/ data/diabetic-retinopath-128-16.tfrecord 128 16

# With labels
python preprocessing.py data/raw_images/ data/diabetic-retinopath-128-16-labeled.tfrecord 128 16 data/labels.csv
```

**What it does:**
- Resizes images to target resolution
- Converts to grayscale
- Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Applies gamma correction (gamma=1.4)
- Performs pixel binning for quantization
- Normalizes to [-1, 1] range
- Compresses with GZIP and saves to TFRecord format

**Expected output:**
```
Processing: 100%|████████████| 1000/1000 [01:23<00:00, 12.03it/s]
Successfully wrote 1000 examples to data/diabetic-retinopath-128-16-labeled.tfrecord
```

---

### 2. Creating Non-IID Data Splits (`create_non_iid_splits.py`)

**Purpose:** Split labeled data into non-IID (non-Independent and Identically Distributed) distributions for federated learning clients.

**Usage:**
Edit the script to configure your input/output paths, then run:
```bash
python create_non_iid_splits.py
```

**Configuration (in script):**
```python
# Input TFRecord
tfrecord_path = 'data/diabetic-retinopath-128-16-labeled.tfrecord'

# Number of clients to create splits for
client_settings = [3, 5, 7, 10]
```

**What it does:**
- Groups data by label (diabetes severity level)
- Randomly assigns labels to 2+ clients
- Ensures each client has minimum 1% of total data
- Redistributes from data-rich to data-poor clients
- Creates separate TFRecord files for each client

**Expected output:**
```
Creating non-IID clusters for 3 clients...
Saved 450 examples to data/train-val-test/non_iid_clusters_3/client_0.tfrecord
Saved 320 examples to data/train-val-test/non_iid_clusters_3/client_1.tfrecord
Saved 230 examples to data/train-val-test/non_iid_clusters_3/client_2.tfrecord
```

---

### 3. Data Validation and Cleanup (`delete_corrupt_images.py`)

**Purpose:** Validate image integrity and remove corrupt files.

**Usage:**
Edit the script to specify your image directory:
```python
folder_path = 'data/raw_images'
```

Then run:
```bash
python delete_corrupt_images.py
```

**What it does:**
- Scans directory for PNG/JPG files
- Validates each image using TensorFlow's decoder
- Moves corrupt files to `backup/` folder
- Reports statistics

**Expected output:**
```
Validating: 100%|████████████| 1000/1000 [00:15<00:00, 65.43it/s]
Validation complete!
Total images: 1000
Valid images: 998
Corrupt images: 2 (moved to backup/)
```

---

## Model Training

### 4. Pretraining (`pretraining.py`)

**Purpose:** Pretrain DCGAN on CT scans for transfer learning.

**Usage:**
Edit the configuration in the script:
```python
CONFIG = {
    "tfrecord_path": "data/rsna-abdominal-128-16.tfrecord",
    "batch_size": 16,
    "latent_dim": 200,
    "learning_rate": 0.0002,
    "beta_1": 0.5,
    "epochs": 1  # Typically 1 epoch for pretraining
}
```

Then run:
```bash
python pretraining.py
```

**What it does:**
- Loads CT scan data from TFRecord
- Trains DCGAN for specified epochs
- Saves generator and discriminator every epoch
- Generates sample images

**Expected output:**
```
Epoch 1/1
1563/1563 [==============================] - 245s 157ms/step - d_loss: 0.6931 - g_loss: 0.6931
Saved generator to federated_learning/models/pretraining/generator_epoch_0001.keras
Saved discriminator to federated_learning/models/pretraining/discriminator_epoch_0001.keras
Generated image saved to federated_learning/generated_images/pretraining/epoch_0001.png
```

**Models saved to:**
- `federated_learning/models/pretraining/generator_epoch_XXXX.keras`
- `federated_learning/models/pretraining/discriminator_epoch_XXXX.keras`

---

### 5. Main Training (`dcgan_training.py`)

**Purpose:** Main training script for both federated and unfederated DCGAN training.

#### Configuration

Edit the `CONFIG` section in the script:

```python
CONFIG = {
    "client_settings": [3, 5, 7, 10],  # Number of clients to test
    "batch_size": 16,
    "latent_dim": 200,
    "learning_rate": 0.0002,
    "beta_1": 0.5,
    "epochs": 5,              # Local epochs per federated round
    "rounds": 2,              # Number of federated rounds
    "images_per_class": 5,    # Images to generate per evaluation
    "generate_interval": 1    # Generate images every N rounds
}
```

#### Usage

**Federated Training:**
```bash
python dcgan_training.py
```

**What it does:**
- Loads non-IID data splits for each client configuration (3, 5, 7, 10 clients)
- For each configuration:
  - Initializes global generator and discriminator
  - For each federated round:
    - Distributes models to clients
    - Each client trains locally for `epochs` iterations
    - Aggregates weights using FedAvg
    - Generates sample images
    - Saves models
- Logs training metrics to TensorBoard

**Expected output:**
```
=== Training with 3 clients ===
Round 1/2
  Training client 0...
    Epoch 1/5 - D Loss: 0.6845, G Loss: 0.7123
    Epoch 2/5 - D Loss: 0.6521, G Loss: 0.7456
    ...
  Training client 1...
  Training client 2...
  Aggregating weights...
  Generated 5 images to federated_learning/generated_images/federated/3_clients/round_001/

Round 2/2
  ...

Models saved to federated_learning/models/federated/3_clients/
```

**Output structure:**
```
federated_learning/
├── models/
│   └── federated/
│       ├── 3_clients/
│       │   ├── global_generator.keras
│       │   └── global_discriminator.keras
│       ├── 5_clients/
│       ├── 7_clients/
│       └── 10_clients/
├── generated_images/
│   └── federated/
│       ├── 3_clients/round_001/
│       ├── 3_clients/round_002/
│       └── ...
└── logs/
    └── tensorboard/
        └── federated_3_clients_20250109_143022/
```

#### Monitoring Training

**TensorBoard:**
```bash
tensorboard --logdir federated_learning/logs/tensorboard
```

Then open http://localhost:6006 in your browser to view:
- Generator loss over time
- Discriminator loss over time
- Generated images per round

---

## Evaluation and Analysis

### 6. Evaluation Metrics (`evaluation_metrics.py`)

**Purpose:** Calculate Fréchet Inception Distance (FID) and Inception Score (IS) for generated images.

**Usage:**
Edit the configuration in the script:
```python
model_base_dir = "federated_learning/models/federated"
tfrecord_path = "data/diabetic-retinopath-128-16-labeled.tfrecord"
client_settings = [3, 5, 7, 10]
```

Then run:
```bash
python evaluation_metrics.py
```

**What it does:**
- Loads real images from TFRecord
- For each client configuration:
  - Loads trained generator
  - Generates synthetic images
  - Calculates FID (lower is better)
  - Calculates Inception Score (higher is better)
- Creates comparison plots
- Saves results

**Expected output:**
```
Evaluating 3 clients...
  Generating 1000 synthetic images...
  Calculating FID... FID: 45.23
  Calculating Inception Score... IS: 2.14 ± 0.08

Evaluating 5 clients...
  FID: 42.67
  IS: 2.28 ± 0.10

Summary:
Client Setting | FID Score | Inception Score
--------------+-----------+----------------
3 clients      | 45.23     | 2.14 ± 0.08
5 clients      | 42.67     | 2.28 ± 0.10
7 clients      | 39.82     | 2.45 ± 0.12
10 clients     | 38.15     | 2.61 ± 0.15

Results saved to federated_learning/evaluation_results/
```

**Interpreting results:**
- **FID (Fréchet Inception Distance):** Measures similarity between real and synthetic distributions. Lower is better. Good FID < 50.
- **Inception Score:** Measures image quality and diversity. Higher is better. Medical images typically score 2-4.

---

### 7. Advanced FID Calculation (`fid_calculator.py`)

**Purpose:** Robust FID calculation with command-line interface and multiple TFRecord format support.

**Usage:**
```bash
python fid_calculator.py \
  --model-dir federated_learning/models/federated \
  --tfrecord-path data/diabetic-retinopath-128-16-labeled.tfrecord \
  --batch-size 16 \
  --num-images 1000 \
  --latent-dim 200
```

**Parameters:**
- `--model-dir`: Base directory containing trained models
- `--tfrecord-path`: Path to real images TFRecord
- `--batch-size`: Batch size for processing (default: 16)
- `--num-images`: Number of images to generate (default: 1000)
- `--latent-dim`: Generator latent dimension (default: 200)

**What it does:**
- Automatically detects client configurations (3, 5, 7, 10)
- Handles multiple TFRecord format variations
- Generates specified number of synthetic images
- Calculates FID score for each configuration
- Provides detailed error handling

**Expected output:**
```
Found client configurations: [3, 5, 7, 10]

Processing 3 clients...
  Loading generator from federated_learning/models/federated/3_clients/global_generator.keras
  Generating 1000 synthetic images...
  Loading 1000 real images from TFRecord...
  Computing FID... FID: 45.23

Processing 5 clients...
  FID: 42.67

...

Summary:
3 clients: FID = 45.23
5 clients: FID = 42.67
7 clients: FID = 39.82
10 clients: FID = 38.15
```

---

### 8. Privacy Evaluation (`privacy_evaluation.py`)

**Purpose:** Comprehensive privacy leakage assessment using membership inference, model inversion, and reconstruction error analysis.

**Usage:**
```bash
python privacy_evaluation.py \
  --model-dir federated_learning/models/federated \
  --data-path data/diabetic-retinopath-128-16-labeled.tfrecord \
  --num-samples 1000 \
  --batch-size 16
```

**Parameters:**
- `--model-dir`: Directory containing trained models
- `--data-path`: Path to training data TFRecord
- `--num-samples`: Number of samples for evaluation (default: 1000)
- `--batch-size`: Batch size for processing (default: 16)

**What it does:**
1. **Membership Inference Attack:**
   - Determines if specific samples were in training data
   - Calculates attack accuracy and ROC curves

2. **Model Inversion Attack:**
   - Attempts to reconstruct original data from model
   - Measures reconstruction quality

3. **Reconstruction Error Analysis:**
   - Quantifies data recovery accuracy
   - Compares training vs. non-training samples

**Expected output:**
```
Privacy Evaluation Report
========================

Client Setting: 3 clients
-------------------------
Membership Inference:
  Attack Accuracy: 52.3%
  AUC-ROC: 0.54
  True Positive Rate: 0.51
  False Positive Rate: 0.48

Model Inversion:
  Reconstruction MSE: 0.023
  SSIM Score: 0.45

Reconstruction Error:
  Training samples MSE: 0.018 ± 0.003
  Non-training samples MSE: 0.029 ± 0.005
  Privacy leakage score: 0.38

Client Setting: 5 clients
-------------------------
  Attack Accuracy: 51.8% (more privacy-preserving)
  ...

Summary:
More clients → Better privacy preservation
Federated learning provides significant privacy benefits

Results saved to privacy_evaluation_results/
```

**Interpreting results:**
- **Attack Accuracy near 50%:** Good privacy (random guessing)
- **Attack Accuracy > 60%:** Privacy concerns
- **Lower Reconstruction MSE:** Potential privacy leakage
- **Higher client count:** Generally better privacy

---

### 9. Statistical Validation (`statistical_validation.py`)

**Purpose:** Multi-run statistical analysis with confidence intervals and significance tests.

**Usage:**
Edit the configuration:
```python
NUM_RUNS = 5  # Number of independent training runs
CONFIG = {
    # Same as dcgan_training.py
}
```

Then run:
```bash
python statistical_validation.py
```

**What it does:**
- Runs complete training pipeline `NUM_RUNS` times
- Collects metrics across runs:
  - Discriminator realism scores
  - FID scores
  - Inception scores
- Calculates statistics:
  - Mean and standard deviation
  - 95% confidence intervals
  - Statistical significance tests (t-tests)
- Generates comparison visualizations

**Expected output:**
```
Statistical Validation - Run 1/5
===============================
Training 3 clients... FID: 45.23, IS: 2.14
Training 5 clients... FID: 42.67, IS: 2.28
...

Statistical Validation - Run 2/5
===============================
...

Final Statistical Summary
=========================

3 Clients:
  FID: 44.82 ± 1.23 (95% CI: [43.21, 46.43])
  IS:  2.16 ± 0.08 (95% CI: [2.05, 2.27])

5 Clients:
  FID: 42.15 ± 1.45 (95% CI: [40.35, 43.95])
  IS:  2.31 ± 0.11 (95% CI: [2.17, 2.45])

Significance Tests:
  3 vs 5 clients FID: p < 0.05 (significant improvement)
  5 vs 7 clients FID: p < 0.05 (significant improvement)
  7 vs 10 clients FID: p = 0.12 (not significant)

Conclusion:
  Optimal client count: 7 (best tradeoff between quality and privacy)

Results saved to federated_learning/statistical_analysis/
```

**Time estimate:**
- Each run: ~30-60 minutes (depends on hardware)
- Total for 5 runs: 2.5-5 hours

---

## Utilities

### 10. Inspect TFRecord with Labels (`tfrecord_inspect_with_labels.py`)

**Purpose:** Visualize labeled TFRecord data for verification.

**Usage:**
```bash
python tfrecord_inspect_with_labels.py data/diabetic-retinopath-128-16-labeled.tfrecord
```

**What it does:**
- Loads first example from TFRecord
- Displays image shape, data range, and label
- Shows sample pixel values

**Expected output:**
```
Inspecting: data/diabetic-retinopath-128-16-labeled.tfrecord

Example 1:
  Shape: (128, 128, 1)
  Data range: [-1.0, 0.98]
  Label: 2
  Sample pixels (top-left 5×5):
    [[-0.85 -0.82 -0.79 -0.76 -0.73]
     [-0.81 -0.78 -0.75 -0.72 -0.69]
     ...]
```

---

### 11. Inspect TFRecord without Labels (`tfrecord_inspect_no_labels.py`)

**Purpose:** Visualize unlabeled TFRecord data.

**Usage:**
```bash
python tfrecord_inspect_no_labels.py data/rsna-abdominal-128-16.tfrecord
```

**What it does:**
- Same as labeled version but without label display
- Useful for CT scan pretraining data

**Expected output:**
```
Inspecting: data/rsna-abdominal-128-16.tfrecord

Example 1:
  Shape: (128, 128, 1)
  Data range: [-1.0, 0.95]
  Sample pixels (top-left 5×5):
    [[-0.90 -0.87 -0.84 -0.81 -0.78]
     ...]
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Errors

**Symptoms:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
- Reduce `batch_size` in CONFIG (try 8 instead of 16)
- Reduce `num_images` in evaluation scripts
- Enable GPU memory growth:
  ```python
  gpus = tf.config.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  ```

#### 2. TFRecord Format Errors

**Symptoms:**
```
DataLossError: corrupted record
InvalidArgumentError: Feature not found
```

**Solutions:**
- Check TFRecord with inspection scripts first
- Verify TFRecord was created successfully
- Use `fid_calculator.py` which handles multiple formats
- Regenerate TFRecord with `preprocessing.py`

#### 3. Model Loading Errors

**Symptoms:**
```
ValueError: Unable to restore custom object
```

**Solutions:**
- Ensure `custom_layers.py` is imported before loading models
- Use absolute paths for model directories
- Check that model files exist at specified paths

#### 4. Slow Training Performance

**Solutions:**
- Enable GPU acceleration (check with `tf.config.list_physical_devices('GPU')`)
- Increase batch size if memory allows
- Use TFRecord format (faster than loading individual images)
- Enable XLA optimization:
  ```python
  tf.config.optimizer.set_jit(True)
  ```

#### 5. Poor Image Quality

**Symptoms:**
- Generated images are blurry or noisy
- FID scores > 100
- Inception scores < 1.5

**Solutions:**
- Train for more federated rounds (increase `rounds` in CONFIG)
- Increase local epochs per round
- Use pretrained weights from `pretraining.py`
- Adjust learning rate (try 0.0001 instead of 0.0002)
- Verify data preprocessing is correct

#### 6. Privacy Evaluation Issues

**Symptoms:**
```
Cannot determine if sample was in training set
```

**Solutions:**
- Ensure you're using the same TFRecord for training and evaluation
- Check that `--data-path` points to correct training data
- Increase `--num-samples` for more reliable statistics

---

## Best Practices

### Data Preparation
1. Always validate images with `delete_corrupt_images.py` before preprocessing
2. Use consistent resolution (128×128) across all experiments
3. Keep original data and preprocessed TFRecords separate
4. Document label mappings in CSV files

### Training
1. Start with pretraining for better convergence
2. Use 3 clients initially to verify pipeline
3. Monitor training with TensorBoard
4. Save models regularly (done automatically)
5. Generate sample images to visually verify quality

### Evaluation
1. Run statistical validation (5+ runs) for publication-quality results
2. Calculate both FID and IS (complementary metrics)
3. Perform privacy evaluation for each configuration
4. Compare federated vs. unfederated baselines

### Experimentation
1. Use git to track configuration changes
2. Document experiments in README or notebook
3. Save all results with descriptive names
4. Keep generated images for qualitative assessment

---

## Quick Start Example

Complete workflow from raw images to trained model:

```bash
# 1. Setup
uv sync
cd /path/to/FedGAN

# 2. Validate data
python delete_corrupt_images.py

# 3. Preprocess images
python preprocessing.py \
  data/raw_images/ \
  data/diabetic-retinopath-128-16-labeled.tfrecord \
  128 16 \
  data/labels.csv

# 4. Create non-IID splits
python create_non_iid_splits.py

# 5. (Optional) Pretrain on CT scans
python pretraining.py

# 6. Train federated model
python dcgan_training.py

# 7. Monitor training
tensorboard --logdir federated_learning/logs/tensorboard

# 8. Evaluate results
python evaluation_metrics.py

# 9. Assess privacy
python privacy_evaluation.py \
  --model-dir federated_learning/models/federated \
  --data-path data/diabetic-retinopath-128-16-labeled.tfrecord

# 10. (Optional) Statistical validation
python statistical_validation.py
```

**Estimated time:**
- Data preparation: 15-30 minutes
- Training (all configurations): 2-4 hours
- Evaluation: 30-60 minutes

---

## Configuration Reference

### Key Parameters

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `batch_size` | 16 | Training batch size | Increase if GPU memory allows |
| `latent_dim` | 200 | Generator input size | 100-200 typical for GANs |
| `learning_rate` | 0.0002 | Optimizer step size | Reduce if training unstable |
| `beta_1` | 0.5 | Adam momentum | 0.5 typical for GANs |
| `epochs` | 5 | Local epochs per round | Increase for better convergence |
| `rounds` | 2 | Federated rounds | Increase for better quality |
| `client_settings` | [3,5,7,10] | Client configurations | Add more for experiments |
| `resolution` | 128 | Image size | 64, 128, or 256 typical |
| `bin_size` | 16 | Pixel quantization | 8, 16, or 32 typical |

### File Paths (Default)

```python
# Data
data/raw_images/                                    # Raw image input
data/diabetic-retinopath-128-16-labeled.tfrecord  # Processed data
data/train-val-test/non_iid_clusters_*/            # Client splits

# Models
federated_learning/models/pretraining/             # Pretrained models
federated_learning/models/federated/               # Federated models
federated_learning/models/unfederated/             # Baseline models

# Output
federated_learning/generated_images/               # Generated samples
federated_learning/evaluation_results/             # Evaluation metrics
federated_learning/logs/tensorboard/               # Training logs
privacy_evaluation_results/                        # Privacy assessment
```

---

## Additional Resources

### Project Files
- `README.md`: Project overview and background
- `LICENSE`: MIT License
- `.gitignore`: Version control exclusions
- `pyproject.toml`: Dependency specifications

### Key Papers
- Federated Learning: McMahan et al. (2017) - "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- DCGAN: Radford et al. (2015) - "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
- FID: Heusel et al. (2017) - "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"

### TensorFlow Documentation
- TFRecord format: https://www.tensorflow.org/tutorials/load_data/tfrecord
- Custom training loops: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
- Keras models: https://www.tensorflow.org/guide/keras/sequential_model

---

## Getting Help

If you encounter issues:

1. Check this INSTRUCTIONS.md document
2. Review error messages carefully
3. Verify file paths and configurations
4. Check data integrity with inspection tools
5. Review the [Troubleshooting](#troubleshooting) section
6. Consult TensorFlow documentation for framework-specific issues

For code-specific questions:
- Check inline comments in Python files
- Review function docstrings
- Examine the codebase structure in README.md

---

## Version Information

- **FedGAN Version**: 0.1.0
- **TensorFlow Version**: 2.17
- **Python Version**: 3.11+
- **Last Updated**: 2025-01-09

---

**Happy training! 🚀**

Generate privacy-preserving synthetic medical images while advancing federated learning research.
