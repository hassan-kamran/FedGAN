# evaluation_metrics.py
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.pyplot as plt
from scipy import linalg
import glob


# Load InceptionV3 model for metrics calculation
def load_inception_model():
    return InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))


# Load images from directory
def load_images(directory, max_images=100):
    image_files = glob.glob(os.path.join(directory, "*.png"))
    images = []
    for img_path in image_files[:max_images]:
        img = plt.imread(img_path)
        if img.ndim == 2:  # If grayscale
            img = np.stack((img,) * 3, axis=-1)  # Convert to RGB
        images.append(img)
    return np.array(images)


# Calculate Inception Score
def calculate_inception_score(images, model, batch_size=50, splits=10):
    # Prepare images for Inception
    processed_images = []
    for img in images:
        # Resize to Inception input size
        img_resized = tf.image.resize(img, (299, 299))
        processed_images.append(img_resized)

    processed_images = np.stack(processed_images)

    # Get predictions
    preds = []
    n_batches = int(np.ceil(float(len(processed_images)) / float(batch_size)))
    for i in range(n_batches):
        batch = processed_images[i * batch_size : (i + 1) * batch_size]
        batch = preprocess_input(batch)  # Scale for InceptionV3
        pred = model.predict(batch)
        preds.append(pred)

    preds = np.concatenate(preds, axis=0)

    # Calculate inception score
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits) : (i + 1) * (len(preds) // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)


# Calculate FID
def calculate_fid(real_images, fake_images, model, batch_size=50):
    # Prepare real images
    real_processed = []
    for img in real_images:
        img_resized = tf.image.resize(img, (299, 299))
        real_processed.append(img_resized)
    real_processed = np.stack(real_processed)

    # Prepare fake images
    fake_processed = []
    for img in fake_images:
        img_resized = tf.image.resize(img, (299, 299))
        fake_processed.append(img_resized)
    fake_processed = np.stack(fake_processed)

    # Get activations
    real_act = get_activations(real_processed, model, batch_size)
    fake_act = get_activations(fake_processed, model, batch_size)

    # Calculate statistics
    mu1, sigma1 = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu2, sigma2 = fake_act.mean(axis=0), np.cov(fake_act, rowvar=False)

    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# Helper function to get activations
def get_activations(images, model, batch_size=50):
    n_batches = int(np.ceil(float(images.shape[0]) / float(batch_size)))
    activations = []
    for i in range(n_batches):
        batch = images[i * batch_size : (i + 1) * batch_size]
        batch = preprocess_input(batch)
        act = model.predict(batch)
        activations.append(act)
    return np.concatenate(activations, axis=0)


# Main evaluation function
def evaluate_all_client_settings():
    # Load inception model
    inception_model = load_inception_model()

    # Define paths
    base_dir = "federated_learning/generated_images"
    real_images_dir = "data/real_samples"  # Directory containing some real samples

    # Create results directory
    results_dir = "metric_results"
    os.makedirs(results_dir, exist_ok=True)

    # Load real images
    real_images = load_images(real_images_dir)

    # Results storage
    results = {"client_count": [], "fid": [], "is_mean": [], "is_std": []}

    # Evaluate each client setting
    for client_count in [3, 5, 7, 10]:
        fake_dir = os.path.join(
            base_dir, f"federated/clusters_{client_count}/epoch_0002_latest"
        )

        if os.path.exists(fake_dir):
            fake_images = load_images(fake_dir)

            if len(fake_images) > 0:
                # Calculate FID
                fid_score = calculate_fid(real_images, fake_images, inception_model)

                # Calculate IS
                is_mean, is_std = calculate_inception_score(
                    fake_images, inception_model
                )

                # Store results
                results["client_count"].append(client_count)
                results["fid"].append(fid_score)
                results["is_mean"].append(is_mean)
                results["is_std"].append(is_std)

                # Save to file
                with open(f"{results_dir}/metrics_{client_count}clients.txt", "w") as f:
                    f.write(f"Client Count: {client_count}\n")
                    f.write(f"FID Score: {fid_score:.4f}\n")
                    f.write(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}\n")

    # Create visualization
    if results["client_count"]:
        plt.figure(figsize=(12, 5))

        # FID plot
        plt.subplot(1, 2, 1)
        plt.bar([f"{c} Clients" for c in results["client_count"]], results["fid"])
        plt.title("Fréchet Inception Distance (lower is better)")
        plt.ylabel("FID Score")

        # IS plot
        plt.subplot(1, 2, 2)
        plt.bar(
            [f"{c} Clients" for c in results["client_count"]],
            results["is_mean"],
            yerr=results["is_std"],
            capsize=5,
        )
        plt.title("Inception Score (higher is better)")
        plt.ylabel("IS Score")

        plt.tight_layout()
        plt.savefig(f"{results_dir}/metrics_comparison.png")

        print("Evaluation complete. Results saved to:", results_dir)


if __name__ == "__main__":
    evaluate_all_client_settings()
