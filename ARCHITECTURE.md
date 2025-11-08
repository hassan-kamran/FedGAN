# FedGAN Architecture Documentation

This document describes the architecture of the refactored FedGAN codebase.

## Design Principles

The refactoring follows these principles:

1. **SOLID Principles**
   - Single Responsibility: Each class has one reason to change
   - Open/Closed: Open for extension, closed for modification
   - Liskov Substitution: Subtypes can replace base types
   - Interface Segregation: Clients shouldn't depend on unused interfaces
   - Dependency Inversion: Depend on abstractions, not concretions

2. **DRY (Don't Repeat Yourself)**
   - Eliminated 6 duplicate TFRecord parsers
   - Consolidated evaluation metrics
   - Unified configuration management

3. **Separation of Concerns**
   - Data loading separate from training
   - Model creation separate from training logic
   - Evaluation separate from training

## Module Overview

### Config Module (`fedgan/config/`)

**Purpose**: Centralized configuration management

**Design Patterns**:
- Builder: `ConfigBuilder` for fluent construction
- Singleton: `ConfigRegistry` for global access

**Key Classes**:
- `ExperimentConfig`: Root configuration
- `ModelConfig`: Model architecture parameters
- `DataConfig`: Data processing parameters
- `FederatedConfig`: Federated learning settings

### Data Module (`fedgan/data/`)

**Purpose**: Unified data loading and preprocessing

**Design Patterns**:
- Strategy: `TFRecordParser` for different formats
- Factory: `DatasetFactory` for dataset creation
- Facade: `FederatedDataLoader` for simple interface
- Chain of Responsibility: Preprocessing pipeline

**Key Classes**:
- `UnlabeledImageParser`, `LabeledImageParser`: Parse TFRecords
- `DatasetFactory`: Creates configured datasets
- `FederatedDataLoader`: Loads client datasets
- `PreprocessingPipeline`: Composable preprocessing

### Models Module (`fedgan/models/`)

**Purpose**: Model architectures and creation

**Design Patterns**:
- Abstract Factory: `ModelFactory` for creating GAN components
- Factory Method: Specific model builders

**Key Classes**:
- `DCGANFactory`: Creates DCGAN generator and discriminator
- `build_generator()`, `build_discriminator()`: Architecture builders
- `ConvLayer`, `TransposedConvLayer`: Custom layers

### Training Module (`fedgan/training/`)

**Purpose**: Training loop infrastructure

**Design Patterns**:
- Template Method: `BaseTrainer` defines skeleton
- Observer: `TrainingCallback` for events

**Key Classes**:
- `BaseTrainer`: Abstract base with template method
- `GANTrainer`: Concrete GAN training implementation
- `TensorBoardCallback`, `ModelCheckpointCallback`: Observers

### Federated Module (`fedgan/federated/`)

**Purpose**: Federated learning orchestration

**Design Patterns**:
- Strategy: `AggregationStrategy` for weight aggregation

**Key Classes**:
- `FederatedTrainer`: Orchestrates federated rounds
- `FedAvgStrategy`, `WeightedFedAvgStrategy`: Aggregation methods

### Evaluation Module (`fedgan/evaluation/`)

**Purpose**: Model evaluation and metrics

**Design Patterns**:
- Strategy: `MetricStrategy` for different metrics
- Facade: `ModelEvaluator` for simple interface

**Key Classes**:
- `FIDMetric`, `InceptionScoreMetric`: Metric strategies
- `ModelEvaluator`: Unified evaluation interface

## Data Flow

```
1. Configuration
   ExperimentConfig.from_yaml()
         ↓
   ConfigRegistry.set_config()

2. Model Creation
   ModelFactory.create_models()
         ↓
   generator, discriminator

3. Data Loading
   FederatedDataLoader.load_client_datasets()
         ↓
   {client_id: tf.data.Dataset}

4. Training
   FederatedTrainer.train_federated()
         ↓
   For each round:
      - Local training on clients
      - Weight aggregation
      - Global model update
         ↓
   Trained models

5. Evaluation
   ModelEvaluator.evaluate()
         ↓
   {metric_name: score}
```

## Extension Points

### Adding a New Model Architecture

1. Create architecture builder in `fedgan/models/architectures.py`
2. Create factory in `fedgan/models/factory.py`
3. Register in `MODEL_FACTORIES`

```python
class MyGANFactory(ModelFactory):
    def create_generator(self):
        return build_my_generator(...)
    
    def create_discriminator(self):
        return build_my_discriminator(...)

MODEL_FACTORIES['megan'] = MyGANFactory
```

### Adding a New Evaluation Metric

1. Implement `MetricStrategy` in `fedgan/evaluation/metrics.py`

```python
class MyMetric(MetricStrategy):
    @property
    def name(self) -> str:
        return "MyMetric"
    
    def compute(self, real_images, fake_images) -> float:
        # Implementation
        return score
```

### Adding a New Aggregation Strategy

1. Implement `AggregationStrategy` in `fedgan/federated/aggregation.py`

```python
class MyAggregation(AggregationStrategy):
    @property
    def name(self) -> str:
        return "my_agg"
    
    def aggregate(self, client_weights):
        # Implementation
        return aggregated_weights

AGGREGATION_STRATEGIES['my_agg'] = MyAggregation
```

## Testing Strategy

### Unit Tests
- Test individual classes in isolation
- Mock dependencies
- Fast execution (<1s per test)

### Integration Tests
- Test interaction between components
- Use real (small) datasets
- Verify end-to-end workflows

### End-to-End Tests
- Full experiment runs
- Marked as `@pytest.mark.slow`
- Run in CI on pull requests

## Performance Considerations

1. **Dataset Caching**: Datasets are cached in memory when possible
2. **TF.function**: Training steps use `@tf.function` for graph optimization
3. **Parallel Data Loading**: `tf.data.AUTOTUNE` for automatic parallelism
4. **Prefetching**: Datasets prefetch batches for GPU utilization

## Security Considerations

1. **Privacy**: No raw data leaves client institutions
2. **Validation**: All configs validated before use
3. **Error Handling**: Graceful failure with informative messages
4. **Type Safety**: Type hints throughout for early error detection

## Future Enhancements

Potential areas for expansion:

1. **Differential Privacy**: Add DP-SGD to local training
2. **Compression**: Gradient compression for communication efficiency
3. **Client Selection**: Sophisticated client sampling strategies
4. **Model Versioning**: Track model evolution across rounds
5. **Experiment Tracking**: Integration with MLflow or Weights & Biases
