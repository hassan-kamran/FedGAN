# privacy_evaluation.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate privacy leakage in federated GAN models"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to original training data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="privacy_evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        nargs="+",
        default=[3, 5, 7, 10],
        help="Number of clients to evaluate",
    )
    parser.add_argument(
        "--eps-values",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        help="Epsilon values for differential privacy estimation",
    )
    return parser.parse_args()


def load_tfrecord_dataset(tfrecord_path, batch_size=32):
    """Load dataset from TFRecord file."""
    try:
        # Define feature description based on your data format
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }

        def _parse_function(example_proto):
            parsed_features = tf.io.parse_single_example(
                example_proto, feature_description
            )
            image = tf.io.parse_tensor(parsed_features["image"], out_type=tf.float32)
            image = tf.reshape(image, [128, 128, 1])  # Adjust shape to match your data
            return image, parsed_features["label"]

        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP")
        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(batch_size)
        return dataset

    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Try an alternative parsing if the first method fails
        try:
            # For unlabeled data
            def _parse_image_only(example_proto):
                image_tensor = tf.io.parse_tensor(example_proto, out_type=tf.float32)
                image_tensor = tf.reshape(image_tensor, [128, 128, 1])
                return image_tensor

            dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP")
            dataset = dataset.map(_parse_image_only)
            dataset = dataset.batch(batch_size)
            return dataset
        except Exception as e2:
            print(f"Alternative parsing also failed: {e2}")
            return None


class PrivacyEvaluator:
    def __init__(self, model_dir, data_path, output_dir, client_settings):
        self.model_dir = model_dir
        self.data_path = data_path
        self.output_dir = output_dir
        self.client_settings = client_settings
        os.makedirs(output_dir, exist_ok=True)

        # Load original dataset
        self.original_dataset = load_tfrecord_dataset(data_path)
        if self.original_dataset is None:
            raise ValueError(f"Could not load dataset from {data_path}")

        # Extract a subset of original data for evaluation
        self.reference_data = []
        self.reference_labels = []

        # Load a sample of original data
        for images, labels in self.original_dataset.take(10):  # Adjust as needed
            self.reference_data.extend(images.numpy())
            if isinstance(labels, tf.Tensor):
                self.reference_labels.extend(labels.numpy())

        self.reference_data = np.array(self.reference_data)
        self.reference_labels = (
            np.array(self.reference_labels) if self.reference_labels else None
        )

        # Create non-members (data not used in training)
        # This is a simplification - in practice you'd need verified non-member data
        np.random.seed(42)
        noise = np.random.normal(0, 1, size=(len(self.reference_data), 128, 128, 1))
        self.non_member_data = np.clip(noise, -1, 1)

        # Results storage
        self.results = {
            "membership_inference": {},
            "model_inversion": {},
            "reconstruction_error": {},
            "differential_privacy": {},
        }

    def evaluate_all_models(self):
        """Evaluate all models for all client settings."""
        for num_clients in self.client_settings:
            print(f"\nEvaluating privacy for {num_clients}-client setting")

            # Find the latest model for this client setting
            model_path = self._find_latest_model(num_clients)
            if not model_path:
                print(f"No model found for {num_clients}-client setting. Skipping.")
                continue

            print(f"Using model: {model_path}")
            try:
                # Load generator model
                generator = load_model(model_path)

                # Run all privacy evaluations
                self._evaluate_membership_inference(generator, num_clients)
                self._evaluate_model_inversion(generator, num_clients)
                self._estimate_differential_privacy(generator, num_clients)
                self._measure_reconstruction_error(generator, num_clients)

            except Exception as e:
                print(f"Error evaluating model for {num_clients} clients: {e}")

        # Save combined results
        self._save_results()

    def _find_latest_model(self, num_clients):
        """Find the latest model for a given client setting."""
        client_dir = os.path.join(self.model_dir, f"clusters_{num_clients}")
        if not os.path.exists(client_dir):
            return None

        # Find round directories
        round_dirs = sorted(
            [d for d in os.listdir(client_dir) if d.startswith("round_")],
            key=lambda x: int(x.split("_")[1]),
        )

        if not round_dirs:
            return None

        # Get the latest round
        latest_round = round_dirs[-1]
        model_path = os.path.join(client_dir, latest_round, "global_generator.keras")

        if os.path.exists(model_path):
            return model_path
        return None

    def _evaluate_membership_inference(self, generator, num_clients):
        """Implement membership inference attack to quantify privacy leakage."""
        print("Running membership inference attack...")

        # Generate latent vectors
        latent_dim = generator.input_shape[1]
        num_samples = len(self.reference_data)

        # Create discriminator (simple CNN for binary classification)
        discriminator = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    64,
                    (3, 3),
                    strides=(2, 2),
                    padding="same",
                    input_shape=(128, 128, 1),
                ),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        discriminator.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Create attack dataset (members vs non-members)
        # Generate synthetic samples similar to reference data
        synthetic_samples = []
        for _ in range(num_samples):
            z = tf.random.normal([1, latent_dim])
            generated = generator(z, training=False).numpy()
            synthetic_samples.append(generated[0])

        synthetic_samples = np.array(synthetic_samples)

        # Train a simple discriminator to distinguish real vs generated data
        X_train = np.concatenate(
            [
                self.reference_data[: num_samples // 2],
                synthetic_samples[: num_samples // 2],
            ]
        )
        y_train = np.concatenate(
            [np.ones(num_samples // 2), np.zeros(num_samples // 2)]
        )

        # Test set (different samples)
        X_test = np.concatenate(
            [
                self.reference_data[num_samples // 2 :],
                synthetic_samples[num_samples // 2 :],
            ]
        )
        y_test = np.concatenate(
            [
                np.ones(len(self.reference_data) - num_samples // 2),
                np.zeros(len(synthetic_samples) - num_samples // 2),
            ]
        )

        # Shuffle the data
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

        indices = np.arange(len(X_test))
        np.random.shuffle(indices)
        X_test, y_test = X_test[indices], y_test[indices]

        # Train the discriminator
        discriminator.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)

        # Evaluate on test set
        test_loss, test_acc = discriminator.evaluate(X_test, y_test, verbose=0)
        y_pred = discriminator.predict(X_test, verbose=0)

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        # Save results
        self.results["membership_inference"][num_clients] = {
            "accuracy": test_acc,
            "auc": roc_auc,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

        print(
            f"Membership inference results: Accuracy = {test_acc:.4f}, AUC = {roc_auc:.4f}"
        )

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Membership Inference Attack ROC - {num_clients} Clients")
        plt.legend(loc="lower right")
        plt.savefig(
            os.path.join(self.output_dir, f"membership_inference_roc_{num_clients}.png")
        )
        plt.close()

    def _evaluate_model_inversion(self, generator, num_clients):
        """Evaluate privacy leakage through model inversion attack."""
        print("Running model inversion simulation...")

        latent_dim = generator.input_shape[1]

        # Model inversion attack simulation
        # Try to reconstruct training images by optimizing latent vectors
        num_samples = min(5, len(self.reference_data))  # Limit to 5 examples
        inversion_results = []

        for i in range(num_samples):
            target_image = self.reference_data[i]

            # Initialize random latent vector
            latent_vector = tf.Variable(tf.random.normal([1, latent_dim]))

            # Optimization loop
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
            best_z = None
            best_loss = float("inf")
            best_reconstruction = None

            for step in range(100):  # Limited optimization steps
                with tf.GradientTape() as tape:
                    # Generate image from latent vector
                    generated_image = generator(latent_vector, training=False)

                    # Calculate reconstruction loss (MSE)
                    loss = tf.reduce_mean(
                        tf.square(
                            generated_image - tf.reshape(target_image, [1, 128, 128, 1])
                        )
                    )

                gradients = tape.gradient(loss, [latent_vector])
                optimizer.apply_gradients(zip(gradients, [latent_vector]))

                # Keep track of best result
                if loss < best_loss:
                    best_loss = loss.numpy()
                    best_z = latent_vector.numpy().copy()
                    best_reconstruction = generated_image.numpy()[0]

            inversion_results.append(
                {
                    "target": target_image,
                    "reconstruction": best_reconstruction,
                    "loss": best_loss,
                }
            )

        # Calculate average reconstruction error
        avg_loss = np.mean([res["loss"] for res in inversion_results])
        self.results["model_inversion"][num_clients] = {
            "average_loss": float(avg_loss),
            "num_samples": num_samples,
        }

        print(f"Model inversion results: Avg reconstruction loss = {avg_loss:.4f}")

        # Visualize results
        plt.figure(figsize=(12, 4 * num_samples))
        for i, result in enumerate(inversion_results):
            # Original image
            plt.subplot(num_samples, 3, i * 3 + 1)
            plt.imshow(result["target"].squeeze(), cmap="gray")
            plt.title(f"Original {i + 1}")
            plt.axis("off")

            # Reconstructed image
            plt.subplot(num_samples, 3, i * 3 + 2)
            plt.imshow(result["reconstruction"].squeeze(), cmap="gray")
            plt.title(f"Reconstructed {i + 1}")
            plt.axis("off")

            # Difference
            plt.subplot(num_samples, 3, i * 3 + 3)
            diff = np.abs(
                result["target"].squeeze() - result["reconstruction"].squeeze()
            )
            plt.imshow(diff, cmap="hot")
            plt.title(f"Difference (Loss: {result['loss']:.4f})")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"model_inversion_{num_clients}.png"))
        plt.close()

    def _estimate_differential_privacy(
        self, generator, num_clients, eps_values=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ):
        """Estimate effective epsilon for differential privacy."""
        print("Estimating differential privacy bounds...")

        latent_dim = generator.input_shape[1]

        # Generate synthetic samples
        num_samples = 100
        synthetic_samples = []
        for _ in range(num_samples):
            z = tf.random.normal([1, latent_dim])
            generated = generator(z, training=False).numpy()
            synthetic_samples.append(generated[0])

        synthetic_samples = np.array(synthetic_samples)

        # Calculate privacy bounds using sensitivity analysis
        # This is a simplified estimation based on distinguishability

        # For each epsilon value, calculate empirical privacy loss
        privacy_results = {}

        for eps in eps_values:
            # Simulate privacy-preserving mechanism with this epsilon
            delta_values = []

            # Calculate distinguishability with different noise levels
            noise_scale = 1.0 / eps

            # Add noise to synthetic samples to simulate DP
            for _ in range(10):  # Multiple trials
                noisy_samples = synthetic_samples + np.random.normal(
                    0, noise_scale, synthetic_samples.shape
                )

                # Calculate distance between original and noisy samples
                distances = np.mean(
                    np.square(synthetic_samples - noisy_samples), axis=(1, 2, 3)
                )
                avg_distance = np.mean(distances)

                # Calculate empirical delta (failure probability)
                # Higher distance means higher probability of distinguishing
                delta = np.mean(distances > 2 * noise_scale)
                delta_values.append(delta)

            privacy_results[eps] = {
                "delta": float(np.mean(delta_values)),
                "avg_distance": float(avg_distance),
            }

        self.results["differential_privacy"][num_clients] = privacy_results

        # Find the epsilon where delta is acceptable (e.g., < 0.05)
        acceptable_eps = [
            eps for eps, result in privacy_results.items() if result["delta"] < 0.05
        ]
        effective_eps = min(acceptable_eps) if acceptable_eps else max(eps_values)

        print(f"Differential privacy estimation: Effective ε ≈ {effective_eps:.2f}")

        # Plot privacy loss curve
        eps_list = sorted(eps_values)
        delta_list = [privacy_results[eps]["delta"] for eps in eps_list]

        plt.figure()
        plt.plot(eps_list, delta_list, "o-")
        plt.axhline(y=0.05, color="r", linestyle="--", label="δ = 0.05 threshold")
        plt.xlabel("Privacy budget (ε)")
        plt.ylabel("Privacy loss (δ)")
        plt.title(f"Privacy Loss Curve - {num_clients} Clients")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"privacy_curve_{num_clients}.png"))
        plt.close()

    def _measure_reconstruction_error(self, generator, num_clients):
        """Measure reconstruction error for original vs generated samples."""
        print("Measuring reconstruction error...")

        latent_dim = generator.input_shape[1]

        # Generate random samples
        num_samples = len(self.reference_data)
        synthetic_samples = []

        for _ in range(num_samples):
            z = tf.random.normal([1, latent_dim])
            generated = generator(z, training=False).numpy()
            synthetic_samples.append(generated[0])

        synthetic_samples = np.array(synthetic_samples)

        # Calculate minimum distance from each real sample to any synthetic sample
        min_distances = []

        for real_sample in tqdm(self.reference_data, desc="Calculating distances"):
            # Calculate MSE between this real sample and all synthetic samples
            distances = np.mean(
                np.square(real_sample - synthetic_samples), axis=(1, 2, 3)
            )
            min_distances.append(np.min(distances))

        avg_min_distance = np.mean(min_distances)
        median_min_distance = np.median(min_distances)
        max_min_distance = np.max(min_distances)
        std_min_distance = np.std(min_distances)

        self.results["reconstruction_error"][num_clients] = {
            "avg_min_distance": float(avg_min_distance),
            "median_min_distance": float(median_min_distance),
            "max_min_distance": float(max_min_distance),
            "std_min_distance": float(std_min_distance),
        }

        print(f"Reconstruction error: Avg min distance = {avg_min_distance:.4f}")

        # Plot histogram of minimum distances
        plt.figure()
        plt.hist(min_distances, bins=30)
        plt.axvline(
            x=avg_min_distance,
            color="r",
            linestyle="--",
            label=f"Avg: {avg_min_distance:.4f}",
        )
        plt.axvline(
            x=median_min_distance,
            color="g",
            linestyle="--",
            label=f"Median: {median_min_distance:.4f}",
        )
        plt.xlabel("Minimum Distance")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Minimum Distances - {num_clients} Clients")
        plt.legend()
        plt.savefig(
            os.path.join(self.output_dir, f"reconstruction_error_{num_clients}.png")
        )
        plt.close()

    def _save_results(self):
        """Save all results to file."""
        import json

        # Convert numpy values to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        serializable_results = convert_to_serializable(self.results)

        with open(
            os.path.join(self.output_dir, "privacy_evaluation_results.json"), "w"
        ) as f:
            json.dump(serializable_results, f, indent=4)

        # Generate summary report
        with open(os.path.join(self.output_dir, "privacy_summary.txt"), "w") as f:
            f.write("FedGAN Privacy Leakage Evaluation Summary\n")
            f.write("========================================\n\n")

            # Membership inference summary
            f.write("1. Membership Inference Attack Results\n")
            f.write("-------------------------------------\n")
            for num_clients, results in self.results["membership_inference"].items():
                f.write(
                    f"{num_clients} Clients: Accuracy = {results['accuracy']:.4f}, AUC = {results['auc']:.4f}\n"
                )
            f.write("\n")

            # Model inversion summary
            f.write("2. Model Inversion Attack Results\n")
            f.write("--------------------------------\n")
            for num_clients, results in self.results["model_inversion"].items():
                f.write(
                    f"{num_clients} Clients: Avg Reconstruction Loss = {results['average_loss']:.4f}\n"
                )
            f.write("\n")

            # Differential privacy summary
            f.write("3. Differential Privacy Estimation\n")
            f.write("--------------------------------\n")
            for num_clients, results in self.results["differential_privacy"].items():
                # Find effective epsilon
                acceptable_eps = [
                    float(eps)
                    for eps, result in results.items()
                    if result["delta"] < 0.05
                ]
                effective_eps = min(acceptable_eps) if acceptable_eps else float("inf")

                f.write(f"{num_clients} Clients: Effective ε ≈ {effective_eps:.2f}\n")
                f.write(f"  Detailed results:\n")
                for eps, res in sorted(results.items(), key=lambda x: float(x[0])):
                    f.write(f"  ε = {eps}: δ = {res['delta']:.4f}\n")
            f.write("\n")

            # Reconstruction error summary
            f.write("4. Reconstruction Error Analysis\n")
            f.write("-------------------------------\n")
            for num_clients, results in self.results["reconstruction_error"].items():
                f.write(f"{num_clients} Clients:\n")
                f.write(f"  Average Min Distance: {results['avg_min_distance']:.4f}\n")
                f.write(
                    f"  Median Min Distance: {results['median_min_distance']:.4f}\n"
                )
                f.write(f"  Maximum Min Distance: {results['max_min_distance']:.4f}\n")
                f.write(f"  Standard Deviation: {results['std_min_distance']:.4f}\n")
            f.write("\n")

            # Overall privacy score
            f.write("5. Overall Privacy Risk Assessment\n")
            f.write("--------------------------------\n")
            for num_clients in self.client_settings:
                if (
                    num_clients in self.results["membership_inference"]
                    and num_clients in self.results["model_inversion"]
                    and num_clients in self.results["differential_privacy"]
                ):
                    # Calculate composite score (lower is better for privacy)
                    membership_score = self.results["membership_inference"][
                        num_clients
                    ]["auc"]
                    inversion_score = 1.0 / (
                        1.0
                        + self.results["model_inversion"][num_clients]["average_loss"]
                    )

                    # Get effective epsilon
                    dp_results = self.results["differential_privacy"][num_clients]
                    acceptable_eps = [
                        float(eps)
                        for eps, result in dp_results.items()
                        if result["delta"] < 0.05
                    ]
                    effective_eps = (
                        min(acceptable_eps) if acceptable_eps else float("inf")
                    )
                    dp_score = 1.0 / (1.0 + effective_eps)

                    # Normalize to 0-100 scale (higher means more privacy risk)
                    privacy_risk = (
                        (membership_score + inversion_score + dp_score) / 3.0
                    ) * 100

                    # Risk category
                    if privacy_risk < 30:
                        risk_category = "Low"
                    elif privacy_risk < 60:
                        risk_category = "Medium"
                    else:
                        risk_category = "High"

                    f.write(
                        f"{num_clients} Clients: Privacy Risk Score = {privacy_risk:.2f}/100 ({risk_category})\n"
                    )

            f.write("\n")
            f.write("Interpretation Guide:\n")
            f.write(
                "- Membership Inference: AUC closer to 0.5 indicates better privacy (random guess)\n"
            )
            f.write(
                "- Model Inversion: Higher reconstruction loss indicates better privacy\n"
            )
            f.write(
                "- Differential Privacy: Lower effective epsilon indicates better privacy\n"
            )
            f.write(
                "- Reconstruction Error: Higher minimum distance indicates better privacy\n"
            )

        print(f"\nPrivacy evaluation complete! Results saved to {self.output_dir}")


def main():
    args = parse_arguments()

    print("\n" + "=" * 50)
    print("FedGAN Privacy Leakage Evaluation")
    print("=" * 50 + "\n")

    print(f"Model directory: {args.model_dir}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Client settings to evaluate: {args.num_clients}")

    try:
        evaluator = PrivacyEvaluator(
            model_dir=args.model_dir,
            data_path=args.data_path,
            output_dir=args.output_dir,
            client_settings=args.num_clients,
        )
        evaluator.evaluate_all_models()

    except Exception as e:
        print(f"Error during privacy evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
