# fid_calculator_all_clusters.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.pyplot as plt
from scipy import linalg
import argparse
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate FID score for GAN models across all clusters"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Base directory containing model folders for all clusters",
    )
    parser.add_argument(
        "--tfrecord-path",
        type=str,
        required=True,
        help="Path to TFRecord file with real images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="fid_results",
        help="Directory to save FID results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for image generation"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to use for FID calculation",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=200, help="Latent dimension for the generator"
    )
    return parser.parse_args()


def _parse_image_from_tfrecord(example_proto):
    """Parse a single image from TFRecord."""
    # Try different parsing approaches
    try:
        # Approach 1: For records with image and label
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }

        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.parse_tensor(parsed_features["image"], out_type=tf.float32)
        if len(tf.shape(image)) < 3:
            image = tf.expand_dims(image, axis=-1)
        return image

    except tf.errors.InvalidArgumentError:
        try:
            # Approach 2: For raw tensor records
            image = tf.io.parse_tensor(example_proto, out_type=tf.float32)
            if len(tf.shape(image)) < 3:
                image = tf.expand_dims(image, axis=-1)
            return image

        except tf.errors.InvalidArgumentError:
            # Approach 3: For simpler image-only format
            try:
                image_tensor = tf.io.decode_raw(example_proto, tf.float32)
                image_tensor = tf.reshape(image_tensor, [128, 128, 1])
                return image_tensor

            except tf.errors.InvalidArgumentError:
                print("Failed to parse TFRecord entry, skipping...")
                return None


def load_images_from_tfrecord(tfrecord_path, num_images=100):
    """Load images from TFRecord file."""
    print(f"Loading real images from: {tfrecord_path}")

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP")

    # Collect images
    real_images = []
    for i, example in enumerate(tqdm(dataset, desc="Loading real images")):
        if i >= num_images:
            break

        try:
            image = _parse_image_from_tfrecord(example)
            if image is not None:
                real_images.append(image.numpy())
        except Exception as e:
            print(f"Error parsing image {i}: {e}")
            continue

    print(f"Successfully loaded {len(real_images)} real images")

    if len(real_images) == 0:
        print("\nDEBUGGING TFRecord parsing:")

        # Try opening the file to confirm it exists and has content
        try:
            with tf.io.gfile.GFile(tfrecord_path, "rb") as f:
                print(f"TFRecord file exists and has size: {f.size()} bytes")
        except Exception as e:
            print(f"Error opening TFRecord file: {e}")

        # Try parsing with explicit error reporting
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP")
        for i, example in enumerate(dataset.take(3)):
            print(f"\nTrying to parse record {i}:")
            # Try all approaches with explicit errors
            try:
                # Try approach 1
                feature_description = {
                    "image": tf.io.FixedLenFeature([], tf.string),
                    "label": tf.io.FixedLenFeature([], tf.int64),
                }
                parsed = tf.io.parse_single_example(example, feature_description)
                print("  Parsing with image+label format: Success")
                img = tf.io.parse_tensor(parsed["image"], out_type=tf.float32)
                print(f"  Image shape: {img.shape}")
            except Exception as e:
                print(f"  Parsing with image+label format: Failed - {e}")

                # Try approach 2
                try:
                    img = tf.io.parse_tensor(example, out_type=tf.float32)
                    print(f"  Parsing as raw tensor: Success, shape={img.shape}")
                except Exception as e2:
                    print(f"  Parsing as raw tensor: Failed - {e2}")

                    # Try approach 3
                    try:
                        img = tf.io.decode_raw(example, tf.float32)
                        print(f"  Parsing with decode_raw: Success, size={img.shape}")
                    except Exception as e3:
                        print(f"  Parsing with decode_raw: Failed - {e3}")

    return np.array(real_images)


def generate_images(generator, num_images, latent_dim, batch_size):
    """Generate synthetic images using the generator model."""
    print(f"Generating {num_images} synthetic images...")

    synthetic_images = []
    num_batches = int(np.ceil(num_images / batch_size))

    for _ in tqdm(range(num_batches), desc="Generating batches"):
        current_batch_size = min(batch_size, num_images - len(synthetic_images))
        z = tf.random.normal([current_batch_size, latent_dim])

        generated_batch = generator(z, training=False)
        synthetic_images.extend(generated_batch.numpy())

        if len(synthetic_images) >= num_images:
            break

    return np.array(synthetic_images[:num_images])


def preprocess_for_inception(images):
    """Preprocess images for InceptionV3 model."""
    # Convert grayscale to RGB if needed
    if images.shape[-1] == 1:
        images = np.concatenate([images, images, images], axis=-1)

    # Resize to inception input size (299x299)
    resized_images = []
    for img in images:
        # Scale from [-1,1] to [0,1] if needed
        if img.min() < 0:
            img = (img + 1) / 2.0
        # Scale from [0,1] to [0,255] if needed
        if img.max() <= 1.0:
            img = img * 255.0

        img = tf.image.resize(img, (299, 299))
        resized_images.append(img)

    # Convert to numpy array and preprocess
    resized_images = np.stack(resized_images)
    return preprocess_input(resized_images)


def calculate_activation_statistics(images, model, batch_size=32):
    """Calculate mean and covariance of InceptionV3 activations."""
    n_batches = int(np.ceil(float(len(images)) / float(batch_size)))
    acts = np.empty((len(images), 2048))

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(images))
        batch = images[start:end]

        pred = model.predict(batch, verbose=0)
        acts[start:end] = pred

    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid(real_images, synthetic_images, inception_model=None):
    """Calculate FID score between real and synthetic images."""
    if inception_model is None:
        print("Loading InceptionV3 model...")
        inception_model = InceptionV3(
            include_top=False, pooling="avg", input_shape=(299, 299, 3)
        )

    print("Preprocessing images for InceptionV3...")
    real_processed = preprocess_for_inception(real_images)
    synthetic_processed = preprocess_for_inception(synthetic_images)

    print("Calculating activation statistics for real images...")
    mu_real, sigma_real = calculate_activation_statistics(
        real_processed, inception_model
    )

    print("Calculating activation statistics for synthetic images...")
    mu_synthetic, sigma_synthetic = calculate_activation_statistics(
        synthetic_processed, inception_model
    )

    print("Calculating FID...")
    fid_value = calculate_frechet_distance(
        mu_real, sigma_real, mu_synthetic, sigma_synthetic
    )
    return fid_value, inception_model


def visualize_images(real_images, synthetic_images, output_path):
    """Visualize real and synthetic images side by side."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Select a subset of images to visualize
    num_viz = min(5, len(real_images), len(synthetic_images))

    plt.figure(figsize=(12, 4 * num_viz))
    for i in range(num_viz):
        # Real image
        plt.subplot(num_viz, 2, 2 * i + 1)
        if real_images[i].shape[-1] == 1:
            plt.imshow(real_images[i].squeeze(), cmap="gray")
        else:
            plt.imshow(real_images[i])
        plt.title(f"Real {i + 1}")
        plt.axis("off")

        # Synthetic image
        plt.subplot(num_viz, 2, 2 * i + 2)
        if synthetic_images[i].shape[-1] == 1:
            plt.imshow(synthetic_images[i].squeeze(), cmap="gray")
        else:
            plt.imshow(synthetic_images[i])
        plt.title(f"Synthetic {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Image comparison saved to {output_path}")


def find_all_cluster_models(model_dir):
    """Find all cluster models in the base directory."""
    cluster_models = []

    # Look for directories matching the pattern "clusters_X"
    for item in os.listdir(model_dir):
        if item.startswith("clusters_") and os.path.isdir(
            os.path.join(model_dir, item)
        ):
            cluster_dir = os.path.join(model_dir, item)

            # Extract cluster number
            try:
                cluster_num = int(item.split("_")[1])
            except (IndexError, ValueError):
                print(f"Warning: Could not parse cluster number from directory: {item}")
                continue

            # Find the latest round
            round_dirs = [
                d
                for d in os.listdir(cluster_dir)
                if d.startswith("round_")
                and os.path.isdir(os.path.join(cluster_dir, d))
            ]

            if not round_dirs:
                print(f"Warning: No round directories found in {cluster_dir}")
                continue

            # Sort by round number
            round_dirs.sort(
                key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0
            )
            latest_round = round_dirs[-1]

            # Check if generator exists
            generator_path = os.path.join(
                cluster_dir, latest_round, "global_generator.keras"
            )
            if os.path.exists(generator_path):
                cluster_models.append(
                    {
                        "cluster_num": cluster_num,
                        "round": latest_round,
                        "model_path": generator_path,
                    }
                )

    # Sort by cluster number
    cluster_models.sort(key=lambda x: x["cluster_num"])
    return cluster_models


def main():
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("FID Score Calculator for FedGAN (All Clusters)")
    print("=" * 50)

    # Find all cluster models
    cluster_models = find_all_cluster_models(args.model_dir)

    if not cluster_models:
        print("ERROR: No cluster models found in the specified directory.")
        print(f"Please check the directory structure: {args.model_dir}")
        return

    print(f"Found {len(cluster_models)} cluster models:")
    for model in cluster_models:
        print(
            f"  Cluster {model['cluster_num']}, {model['round']}: {model['model_path']}"
        )

    # Load real images from TFRecord (do this once)
    real_images = load_images_from_tfrecord(args.tfrecord_path, args.num_images)

    if len(real_images) == 0:
        print("ERROR: No real images could be loaded from the TFRecord file.")
        print("Please check the file format and path and try again.")
        return

    # Load InceptionV3 model (do this once)
    print("Loading InceptionV3 model...")
    inception_model = InceptionV3(
        include_top=False, pooling="avg", input_shape=(299, 299, 3)
    )

    # Results storage
    fid_results = []

    # Process each cluster model
    for model_info in cluster_models:
        cluster_num = model_info["cluster_num"]
        round_name = model_info["round"]
        model_path = model_info["model_path"]

        print("\n" + "=" * 50)
        print(f"Evaluating Cluster {cluster_num}, {round_name}")
        print("=" * 50)

        try:
            # Load generator model
            print(f"Loading generator model from: {model_path}")
            generator = load_model(model_path)

            # Generate synthetic images
            synthetic_images = generate_images(
                generator=generator,
                num_images=len(real_images),  # Match number of real images
                latent_dim=args.latent_dim,
                batch_size=args.batch_size,
            )

            # Visualize some real and synthetic images
            viz_path = os.path.join(
                args.output_dir, f"image_comparison_cluster_{cluster_num}.png"
            )
            visualize_images(real_images, synthetic_images, viz_path)

            # Calculate FID score (reuse inception model)
            fid_value, _ = calculate_fid(real_images, synthetic_images, inception_model)

            # Save result
            fid_results.append(
                {
                    "cluster_num": cluster_num,
                    "round": round_name,
                    "fid_score": fid_value,
                }
            )

            print(f"FID Score for Cluster {cluster_num}: {fid_value:.4f}")

        except Exception as e:
            print(f"Error processing model for Cluster {cluster_num}: {e}")
            import traceback

            traceback.print_exc()

    # Generate comparative report
    if fid_results:
        print("\n" + "=" * 50)
        print("FID Score Comparison Across Clusters")
        print("=" * 50)

        # Sort by FID score (lower is better)
        fid_results.sort(key=lambda x: x["fid_score"])

        for result in fid_results:
            print(f"Cluster {result['cluster_num']}: {result['fid_score']:.4f}")

        # Save results to file
        with open(os.path.join(args.output_dir, "fid_comparison.txt"), "w") as f:
            f.write("FID Score Comparison Across Clusters\n")
            f.write("==================================\n\n")

            for result in fid_results:
                f.write(
                    f"Cluster {result['cluster_num']} ({result['round']}): {result['fid_score']:.4f}\n"
                )

            f.write(
                "\nNote: Lower FID scores indicate better image quality and diversity.\n"
            )

        # Create bar chart visualization
        plt.figure(figsize=(10, 6))
        cluster_nums = [result["cluster_num"] for result in fid_results]
        fid_scores = [result["fid_score"] for result in fid_results]

        bars = plt.bar(cluster_nums, fid_scores)

        # Add FID values above bars
        for i, (bar, score) in enumerate(zip(bars, fid_scores)):
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.1,
                f"{score:.2f}",
                ha="center",
                va="bottom",
            )

        plt.xlabel("Number of Clients")
        plt.ylabel("FID Score (lower is better)")
        plt.title("FID Scores Across Different Client Configurations")
        plt.xticks(cluster_nums)

        # Add grid lines for readability
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "fid_comparison.png"))
        print(
            f"FID comparison chart saved to {os.path.join(args.output_dir, 'fid_comparison.png')}"
        )

    print("\nAll cluster evaluations complete!")


if __name__ == "__main__":
    main()
