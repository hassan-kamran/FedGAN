# FedGAN Migration Guide

This guide helps you migrate from the old codebase structure to the new refactored architecture.

## Overview of Changes

The refactoring introduces:
- **Modular package structure** replacing flat file organization
- **Design patterns** for better code organization
- **Configuration management** replacing hardcoded values
- **Consolidated data loading** replacing 6 duplicate parsers
- **Clean training abstractions** replacing monolithic training scripts

## Quick Start

### Old Way (Before Refactoring)

```python
# Old: Hardcoded config dictionary
CONFIG = {
    'latent_dim': 200,
    'batch_size': 16,
    'learning_rate': 0.0002,
    # ... many more hardcoded values
}

# Old: Direct model creation
from model_dcgan import build_generator, build_discriminator
generator = build_generator(latent_dim=200)
discriminator = build_discriminator()

# Old: Manual data loading with duplicate parsers
def _parse_image_function(example_proto):
    # Duplicated 6 times across different files!
    ...
```

### New Way (After Refactoring)

```python
# New: Load config from YAML
from fedgan.config import ExperimentConfig
config = ExperimentConfig.from_yaml("experiments/fedgan_retinopathy.yaml")

# New: Use factory to create models
from fedgan.models import get_model_factory
factory = get_model_factory('dcgan', config.model, image_size=128)
generator, discriminator = factory.create_models()

# New: Use data loader (no duplication!)
from fedgan.data import FederatedDataLoader
loader = FederatedDataLoader(config)
client_datasets = loader.load_client_datasets(num_clients=5)
```

## Migration Steps

### Step 1: Create Configuration File

**Old code:**
```python
CONFIG = {
    'latent_dim': 200,
    'learning_rate': 0.0002,
    'batch_size': 16,
    # etc...
}
```

**New approach:**
```yaml
# experiments/my_experiment.yaml
name: my_fedgan_experiment
description: My experiment description

model:
  latent_dim: 200
  learning_rate: 0.0002

data:
  batch_size: 16
  image_size: 128

federated:
  num_clients: 5
  local_epochs: 5
  federated_rounds: 10
```

**Load in code:**
```python
from fedgan.config import ExperimentConfig
config = ExperimentConfig.from_yaml("experiments/my_experiment.yaml")
```

### Step 2: Replace Model Creation

**Old code:**
```python
from model_dcgan import build_generator, build_discriminator

generator = build_generator(latent_dim=200)
discriminator = build_discriminator(image_shape=(128, 128, 1))
```

**New code:**
```python
from fedgan.models import DCGANFactory

factory = DCGANFactory(config.model, image_size=128, channels=1)
generator = factory.create_generator()
discriminator = factory.create_discriminator()

# Or create both at once:
generator, discriminator = factory.create_models()
```

### Step 3: Replace Data Loading

**Old code (duplicated everywhere):**
```python
def _parse_image_function(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(parsed['image'], tf.float32)
    image = tf.reshape(image, [128, 128, 1])
    return image

dataset = tf.data.TFRecordDataset('data.tfrecord', compression_type='GZIP')
dataset = dataset.map(_parse_image_function)
dataset = dataset.batch(16)
# etc...
```

**New code (one line):**
```python
from fedgan.data import FederatedDataLoader

loader = FederatedDataLoader(config)

# Load all client datasets
client_datasets = loader.load_client_datasets(num_clients=5)

# Or load specific client
client_0_data = loader.load_client_dataset(client_id=0, num_clients=5)

# Or load centralized datasets
val_data = loader.load_validation_dataset()
```

### Step 4: Replace Preprocessing

**Old code:**
```python
# Preprocessing scattered across files
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(image)

# Gamma correction
inv_gamma = 1.0 / 1.5
table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
corrected = cv2.LUT(enhanced, table)

# Binning
binned = (corrected // 16) * 16

# Normalization
normalized = (binned.astype(np.float32) / 127.5) - 1.0
```

**New code (composable pipeline):**
```python
from fedgan.data import PreprocessingPipeline

# Use preset pipeline
pipeline = PreprocessingPipeline.for_retinopathy(image_size=128)
processed = pipeline(raw_image)

# Or build custom pipeline
from fedgan.data import CLAHEStep, GammaCorrectionStep, NormalizationStep

pipeline = CLAHEStep(
    clip_limit=2.0,
    next_step=GammaCorrectionStep(
        gamma=1.5,
        next_step=NormalizationStep(output_range=(-1, 1))
    )
)
processed = pipeline(raw_image)
```

### Step 5: Replace Training Code

**Old code (monolithic):**
```python
# 900+ lines in dcgan_training.py
def train_federated_dcgan(...):
    # Manual federated training loop
    for round in range(num_rounds):
        for client_id in range(num_clients):
            # Load data
            # Create models
            # Train
            # Collect weights
        # Aggregate weights manually
        # Update global models
```

**New code (clean abstractions):**
```python
from fedgan.federated import FederatedTrainer
from fedgan.models import get_model_factory

# Create trainer
factory = get_model_factory('dcgan', config.model, image_size=128)
trainer = FederatedTrainer(config, factory)

# Load data
loader = FederatedDataLoader(config)
client_datasets = loader.load_client_datasets(num_clients=5)

# Train (all complexity hidden)
history = trainer.train_federated(
    client_datasets,
    rounds=10
)

# Get trained models
models = trainer.get_global_models()
generator = models['generator']
```

### Step 6: Replace Evaluation

**Old code (duplicated across files):**
```python
# FID calculation duplicated in evaluation_metrics.py and fid_calculator.py
def calculate_fid(...):
    # 100+ lines of FID calculation
    ...

def calculate_inception_score(...):
    # More duplicated code
    ...
```

**New code (Strategy pattern):**
```python
from fedgan.evaluation import ModelEvaluator, FIDMetric, InceptionScoreMetric

# Create evaluator with metrics
evaluator = ModelEvaluator([
    FIDMetric(),
    InceptionScoreMetric()
])

# Evaluate
scores = evaluator.evaluate(
    generator=generator,
    real_dataset=val_dataset,
    num_samples=5000,
    latent_dim=200
)

# Results: {'FID': 45.2, 'IS': 2.8}
print(f"FID: {scores['FID']:.2f}")
```

## Complete Example

Here's a complete example showing the new workflow:

```python
from fedgan.config import ExperimentConfig
from fedgan.data import FederatedDataLoader
from fedgan.models import get_model_factory
from fedgan.federated import FederatedTrainer
from fedgan.evaluation import ModelEvaluator, FIDMetric

# 1. Load configuration
config = ExperimentConfig.from_yaml("experiments/fedgan_retinopathy.yaml")

# 2. Create models
factory = get_model_factory('dcgan', config.model, image_size=128)

# 3. Load data
loader = FederatedDataLoader(config)
client_datasets = loader.load_client_datasets(num_clients=5)
val_dataset = loader.load_validation_dataset()

# 4. Train
trainer = FederatedTrainer(config, factory)
history = trainer.train_federated(client_datasets, rounds=10)

# 5. Evaluate
evaluator = ModelEvaluator([FIDMetric()])
models = trainer.get_global_models()
scores = evaluator.evaluate(
    generator=models['generator'],
    real_dataset=val_dataset
)

print(f"Training complete! FID: {scores['FID']:.2f}")

# 6. Save
trainer.save_global_models("outputs/final_models")
```

## Benefits of New Architecture

| Aspect | Old | New |
|--------|-----|-----|
| **Config** | Hardcoded in code | YAML files |
| **Data Loading** | 6 duplicate parsers | 1 unified parser |
| **Models** | Direct creation | Factory pattern |
| **Training** | 900-line monolith | Clean abstractions |
| **Evaluation** | Duplicated code | Strategy pattern |
| **Testability** | Hard to test | Fully testable |
| **Extensibility** | Requires editing core files | Plugin architecture |

## Backward Compatibility

The old files are still present in the repository. You can run the old code as-is. However, we recommend migrating to the new architecture for better maintainability.

## Common Pitfalls

1. **Forgetting to create config**: Always load or create an `ExperimentConfig` first
2. **Mixing old and new**: Don't import from both old files and new `fedgan` package
3. **Wrong data paths**: Update paths in YAML configs to match your data location

## Getting Help

- Check the examples in `experiments/` directory
- Run tests with `make test` to see usage examples
- Review individual module documentation in `fedgan/` subpackages

## Next Steps

1. Create your experiment YAML config
2. Run a quick test with the new architecture
3. Gradually migrate your custom code
4. Add tests for your specific use cases
