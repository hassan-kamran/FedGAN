#!/usr/bin/env python3
"""
Example script showing how to use the refactored FedGAN architecture.

This demonstrates the complete workflow from configuration to evaluation.
"""
from pathlib import Path

from fedgan.config import ConfigBuilder, ExperimentConfig
from fedgan.data import FederatedDataLoader
from fedgan.evaluation import FIDMetric, ModelEvaluator
from fedgan.federated import FederatedTrainer
from fedgan.models import get_model_factory
from fedgan.training.callbacks import ImageGenerationCallback, TensorBoardCallback


def main():
    """Run example federated GAN training."""
    
    # Option 1: Build config programmatically
    config = (
        ConfigBuilder()
        .for_experiment("example_federated_gan")
        .with_description("Example showing new architecture")
        .with_model_config(latent_dim=200, learning_rate=0.0002)
        .with_data_config(image_size=128, batch_size=16)
        .with_federated_config(num_clients=5, local_epochs=5, federated_rounds=2)
        .build()
    )
    
    # Option 2: Load from YAML (recommended)
    # config = ExperimentConfig.from_yaml("experiments/fedgan_retinopathy.yaml")
    
    print(f"Experiment: {config.name}")
    print(f"Model: {config.model.latent_dim}D latent space")
    print(f"Federated: {config.federated.num_clients} clients, "
          f"{config.federated.federated_rounds} rounds\n")
    
    # Create output directories
    config.paths.create_directories()
    
    # Step 1: Create model factory
    print("Creating models...")
    factory = get_model_factory(
        'dcgan',
        config.model,
        image_size=config.data.image_size,
        channels=config.data.channels
    )
    
    # Step 2: Load data
    print("Loading federated datasets...")
    loader = FederatedDataLoader(config)
    
    try:
        client_datasets = loader.load_client_datasets(
            num_clients=config.federated.num_clients
        )
        print(f"Loaded {len(client_datasets)} client datasets")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nThis example requires pre-partitioned data.")
        print("Run create_non_iid_splits.py first to create client data splits.")
        return
    
    # Step 3: Create federated trainer
    print("\nInitializing federated trainer...")
    trainer = FederatedTrainer(config, factory)
    
    # Add callbacks
    generator = trainer.global_generator
    callbacks = [
        TensorBoardCallback(config.paths.log_dir),
        ImageGenerationCallback(
            generator,
            config.paths.output_dir / "images",
            latent_dim=config.model.latent_dim
        )
    ]
    
    # Note: To add callbacks to federated training, we'd need to modify
    # the local trainers. For now, callbacks work with GANTrainer directly.
    
    # Step 4: Train
    print(f"\nStarting federated training for {config.federated.federated_rounds} rounds...")
    history = trainer.train_federated(
        client_datasets,
        rounds=config.federated.federated_rounds
    )
    
    print("\nTraining complete!")
    
    # Step 5: Save models
    print("\nSaving global models...")
    save_dir = config.paths.model_dir / "final"
    trainer.save_global_models(str(save_dir))
    print(f"Models saved to {save_dir}")
    
    # Step 6: Evaluate (if validation data available)
    print("\nEvaluation...")
    try:
        val_dataset = loader.load_validation_dataset()
        
        evaluator = ModelEvaluator([FIDMetric()])
        models = trainer.get_global_models()
        
        scores = evaluator.evaluate(
            generator=models['generator'],
            real_dataset=val_dataset,
            num_samples=1000,  # Reduced for speed
            latent_dim=config.model.latent_dim
        )
        
        print(f"\nFID Score: {scores['FID']:.2f}")
        
        # Save results
        evaluator.save_results(config.paths.output_dir / "evaluation.json")
        
    except FileNotFoundError:
        print("Validation data not found, skipping evaluation")
    
    print("\nDone! Check outputs in:", config.paths.output_dir)


if __name__ == "__main__":
    main()
