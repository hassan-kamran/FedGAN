import os
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.random import normal
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from model_dcgan import build_generator, build_discriminator

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("dcgan_federated_experiment.log"),
    ],
)
logger = logging.getLogger(__name__)


# Loss Functions
def generator_loss(fake_output):
    return tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)
    )


def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            tf.ones_like(real_output) * 0.9, real_output
        )
    )
    fake_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    )
    return real_loss + fake_loss


# Directory Setup
BASE_DIR = os.path.abspath("federated_learning")
DIRS = [
    "models/pretraining/generator",
    "models/pretraining/discriminator",
    "models/federated",
    "models/unfederated",
    "logs/tensorboard",
    "generated_images",
    "evaluation_results",
]

logger.info("\n%s\nDirectory Creation\n%s", "=" * 40, "=" * 40)
for dir_path in DIRS:
    full_path = os.path.join(BASE_DIR, dir_path)
    os.makedirs(full_path, exist_ok=True)
    logger.info("Created/Verified: %s", full_path)

# Global Configuration
CONFIG = {
    "client_settings": [3, 5, 7, 10],
    "base_data_path": os.path.abspath("data/train-val-test"),
    "unfederated_data_path": os.path.abspath(
        "data/diabetic-retinopath-128-16-labeled.tfrecord"
    ),
    "base_models_path": os.path.join(BASE_DIR, "models/federated"),
    "unfederated_models_path": os.path.join(BASE_DIR, "models/unfederated"),
    "generated_images_path": os.path.join(BASE_DIR, "generated_images"),
    "eval_results_path": os.path.join(BASE_DIR, "evaluation_results"),
    "batch_size": 16,
    "latent_dim": 200,
    "learning_rate": 0.0002,
    "beta_1": 0.5,
    "epochs": 5,  # number of local epochs per round (or per unfederated training)
    "rounds": 2,  # number of federated rounds
    "images_per_class": 5,
    "generate_interval": 1,
    "pretrained_generator_path": os.path.abspath(
        "models/pretrained/generator/generator_epoch_0001.keras"
    ),
    "pretrained_discriminator_path": os.path.abspath(
        "models/pretrained/discriminator/discriminator_epoch_0001.keras"
    ),
}


# TensorBoard Setup (if needed)
def setup_tensorboard(log_name):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(BASE_DIR, "logs/tensorboard", log_name, current_time)
    return tf.summary.create_file_writer(log_dir)


# Utility Functions
def verify_paths():
    logger.info("\n%s\nPath Verification\n%s", "=" * 40, "=" * 40)
    logger.info("Current working directory: %s", os.getcwd())
    critical_paths = {
        "BASE_DIR": BASE_DIR,
        "Unfederated Data": CONFIG["unfederated_data_path"],
        "Base Data Directory": CONFIG["base_data_path"],
        "Pretrained Generator": CONFIG["pretrained_generator_path"],
        "Pretrained Discriminator": CONFIG["pretrained_discriminator_path"],
    }
    for name, path in critical_paths.items():
        exists = os.path.exists(path)
        status = "✅ EXISTS" if exists else "❌ MISSING"
        logger.info("%s: %s (%s)", name.ljust(25), path, status)
        if not exists and name == "Unfederated Data":
            logger.info("TROUBLESHOOTING TIPS:")
            logger.info("1. Verify the file exists at: %s", path)
            logger.info("2. Check filename spelling (retinopathy vs retinopath)")
            logger.info("3. Ensure the file is not empty")
            logger.info("4. Confirm directory structure matches.")


def _parse_image_function(example_proto):
    feature_description = {"image": tf.io.FixedLenFeature([], tf.string)}
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    image_tensor = tf.io.parse_tensor(parsed_example["image"], out_type=tf.float32)
    image_tensor = tf.reshape(image_tensor, [128, 128, 1])
    return image_tensor


def create_dataset_from_tfrecord(tfrecord_path, batch_size):
    if not os.path.exists(tfrecord_path):
        raise FileNotFoundError(
            f"TFRecord not found at: {os.path.abspath(tfrecord_path)}"
        )
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP")
    dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
    return (
        dataset.shuffle(1000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def get_dataset_length(tfrecord_path):
    """Calculate the exact number of samples in a TFRecord file"""
    count = 0
    for _ in tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP"):
        count += 1
    return count


def average_weights(weight_list):
    new_weights = []
    for layer_idx in range(len(weight_list[0])):
        layer_weights = [w[layer_idx] for w in weight_list]
        layer_mean = tf.reduce_mean(tf.stack(layer_weights, axis=0), axis=0)
        new_weights.append(layer_mean.numpy())
    return new_weights


def generate_and_save_images(
    generator, epoch, output_dir, num_clusters=None, cluster_id=None
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch:04d}_{timestamp}")
    os.makedirs(epoch_dir, exist_ok=True)
    noise = normal([CONFIG["images_per_class"], CONFIG["latent_dim"]])
    generated_images = generator(noise, training=False)
    for i, img in enumerate(generated_images):
        # Rescale images from [-1,1] to [0,255]
        img = (img * 127.5 + 127.5).numpy().astype("uint8")
        img_path = os.path.join(epoch_dir, f"image_{i + 1}.png")
        plt.imsave(img_path, img.squeeze(), cmap="gray")
    logger.info("Generated images saved to: %s", epoch_dir)


# Training Components
class LocalTrainer:
    def __init__(self, latent_dim, learning_rate, beta_1):
        self.generator = build_generator(latent_dim)
        self.discriminator = build_discriminator()
        # Build models by running a dummy input through them
        _ = self.generator(tf.zeros([1, latent_dim]))
        _ = self.discriminator(tf.zeros([1, 128, 128, 1]))
        self.gen_optimizer = Adam(learning_rate, beta_1=beta_1)
        self.disc_optimizer = Adam(learning_rate, beta_1=beta_1)

    @tf.function(reduce_retracing=True)
    def train_step(self, images):
        noise = normal([tf.shape(images)[0], CONFIG["latent_dim"]])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        return gen_loss, disc_loss


# Unfederated Training Function with Loss Tracking & Visualization
def train_unfederated_dcgan(tfrecord_path):
    logger.info("\n%s\nStarting Unfederated Training\n%s", "=" * 40, "=" * 40)
    writer = setup_tensorboard("unfederated")
    global_step = tf.Variable(0, dtype=tf.int64)
    try:
        dataset = create_dataset_from_tfrecord(tfrecord_path, CONFIG["batch_size"])
    except Exception as e:
        logger.error("Dataset Error: %s", str(e))
        return None, None

    generator = build_generator(CONFIG["latent_dim"])
    discriminator = build_discriminator()
    _ = generator(tf.zeros([1, CONFIG["latent_dim"]]))
    _ = discriminator(tf.zeros([1, 128, 128, 1]))
    gen_optimizer = Adam(CONFIG["learning_rate"], beta_1=CONFIG["beta_1"])
    disc_optimizer = Adam(CONFIG["learning_rate"], beta_1=CONFIG["beta_1"])

    # Determine number of batches per epoch based on the dataset length
    num_samples = get_dataset_length(tfrecord_path)
    batches_per_epoch = num_samples // CONFIG["batch_size"]
    loss_history_unfed = {"gen": [], "disc": []}
    dataset_iter = iter(dataset)

    for epoch in range(CONFIG["epochs"]):
        logger.info("Epoch %d/%d", epoch + 1, CONFIG["epochs"])
        epoch_gen_loss_sum = 0.0
        epoch_disc_loss_sum = 0.0

        for batch_idx in range(batches_per_epoch):
            try:
                batch_images = next(dataset_iter)
            except StopIteration:
                dataset_iter = iter(dataset)
                batch_images = next(dataset_iter)
            noise = normal([CONFIG["batch_size"], CONFIG["latent_dim"]])
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)
                real_output = discriminator(batch_images, training=True)
                fake_output = discriminator(generated_images, training=True)
                g_loss = generator_loss(fake_output)
                d_loss = discriminator_loss(real_output, fake_output)
            gen_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
            disc_gradients = disc_tape.gradient(
                d_loss, discriminator.trainable_variables
            )
            gen_optimizer.apply_gradients(
                zip(gen_gradients, generator.trainable_variables)
            )
            disc_optimizer.apply_gradients(
                zip(disc_gradients, discriminator.trainable_variables)
            )
            epoch_gen_loss_sum += g_loss.numpy()
            epoch_disc_loss_sum += d_loss.numpy()

            with writer.as_default():
                tf.summary.scalar("generator_loss", g_loss, step=global_step)
                tf.summary.scalar("discriminator_loss", d_loss, step=global_step)
                if batch_idx % 10 == 0:
                    tf.summary.image(
                        "generated_images",
                        generated_images,
                        max_outputs=4,
                        step=global_step,
                    )
            global_step.assign_add(1)
            if batch_idx % 10 == 0:
                logger.info(
                    "Batch %d | G Loss: %.4f | D Loss: %.4f",
                    batch_idx,
                    g_loss.numpy(),
                    d_loss.numpy(),
                )

        avg_gen_loss = epoch_gen_loss_sum / batches_per_epoch
        avg_disc_loss = epoch_disc_loss_sum / batches_per_epoch
        loss_history_unfed["gen"].append(avg_gen_loss)
        loss_history_unfed["disc"].append(avg_disc_loss)
        logger.info(
            "Epoch %d Average Generator Loss: %.4f | Discriminator Loss: %.4f",
            epoch + 1,
            avg_gen_loss,
            avg_disc_loss,
        )

        # Save models after each epoch
        generator_path = os.path.join(
            CONFIG["unfederated_models_path"],
            f"unfederated_generator_epoch_{epoch + 1}.keras",
        )
        discriminator_path = os.path.join(
            CONFIG["unfederated_models_path"],
            f"unfederated_discriminator_epoch_{epoch + 1}.keras",
        )
        generator.save(generator_path)
        discriminator.save(discriminator_path)
        logger.info("Saved unfederated models for epoch %d", epoch + 1)
        generate_and_save_images(
            generator,
            epoch + 1,
            os.path.join(CONFIG["generated_images_path"], "unfederated"),
        )

    # Plot Unfederated Loss Curves
    plt.figure(figsize=(10, 6))
    epochs_range = list(range(1, CONFIG["epochs"] + 1))
    plt.plot(
        epochs_range, loss_history_unfed["gen"], marker="o", label="Generator Loss"
    )
    plt.plot(
        epochs_range, loss_history_unfed["disc"], marker="o", label="Discriminator Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Unfederated Training Loss Curves")
    plt.legend()
    unfed_plot_path = os.path.join(CONFIG["eval_results_path"], "loss_unfederated.png")
    plt.tight_layout()
    plt.savefig(unfed_plot_path)
    plt.close()
    logger.info("Saved unfederated loss plot to: %s", unfed_plot_path)

    return generator, discriminator


# Evaluation Function (kept for saving model outputs)
def evaluate_generator(generator, discriminator, num_clusters, round_num):
    logger.info("Evaluating %d-cluster model from round %d", num_clusters, round_num)
    noise = normal([CONFIG["images_per_class"], CONFIG["latent_dim"]])
    generated_images = generator(noise, training=False)
    predictions = discriminator(generated_images)
    avg_score = tf.reduce_mean(predictions).numpy()
    eval_dir = os.path.join(CONFIG["eval_results_path"], f"clusters_{num_clusters}")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, f"round_{round_num}_scores.txt"), "w") as f:
        f.write(f"Average Realism Score: {avg_score:.4f}\n")
        for i, score in enumerate(predictions.numpy()):
            f.write(f"Image {i + 1}: {score[0]:.4f}\n")
    # Save evaluation image grid
    plt.figure(figsize=(10, 10))
    for i in range(CONFIG["images_per_class"]):
        plt.subplot(1, CONFIG["images_per_class"], i + 1)
        plt.imshow((generated_images[i].numpy().squeeze() + 1) * 127.5, cmap="gray")
        plt.title(f"Score: {predictions[i][0]:.2f}")
        plt.axis("off")
    plt.tight_layout()
    eval_img_path = os.path.join(eval_dir, f"round_{round_num}_eval_images.png")
    plt.savefig(eval_img_path)
    plt.close()
    logger.info("Evaluation results saved to: %s", eval_img_path)
    return avg_score


# Federated Training Function with Loss Logging and Visualization
def train_federated_dcgan(unfed_discriminator):
    logger.info("\n%s\nStarting Federated Training\n%s", "=" * 40, "=" * 40)
    # Dictionary to store per-client loss history for each setting.
    # Structure: { client_setting: { client_id: { "gen": [losses...], "disc": [losses...] } } }
    loss_history = {}

    for num_clusters in CONFIG["client_settings"]:
        logger.info("Processing %d-cluster scenario", num_clusters)
        loss_history[num_clusters] = {}
        # Initialize loss history for each client (cluster)
        for cluster_id in range(1, num_clusters + 1):
            loss_history[num_clusters][cluster_id] = {"gen": [], "disc": []}
        scenario_writer = setup_tensorboard(f"federated/clusters_{num_clusters}")
        global_step = tf.Variable(0, dtype=tf.int64)
        global_generator = build_generator(CONFIG["latent_dim"])
        global_discriminator = build_discriminator()
        _ = global_generator(tf.zeros([1, CONFIG["latent_dim"]]))
        _ = global_discriminator(tf.zeros([1, 128, 128, 1]))

        for round_num in range(CONFIG["rounds"]):
            logger.info(
                "Starting Round %d/%d for %d-cluster scenario",
                round_num + 1,
                CONFIG["rounds"],
                num_clusters,
            )
            round_writer = setup_tensorboard(
                f"federated/clusters_{num_clusters}/round_{round_num + 1}"
            )
            local_g_weights = []
            local_d_weights = []

            # Process each client (cluster)
            for cluster_id in range(1, num_clusters + 1):
                try:
                    split_path = os.path.join(
                        CONFIG["base_data_path"],
                        f"non_iid_clusters_{num_clusters}",
                        f"non_iid_split_{cluster_id}.tfrecord",
                    )
                    if not os.path.exists(split_path):
                        logger.warning("Missing split: %s", split_path)
                        continue
                    num_samples = get_dataset_length(split_path)
                    if num_samples == 0:
                        logger.warning("Empty split: %s", split_path)
                        continue
                    dataset = create_dataset_from_tfrecord(
                        split_path, CONFIG["batch_size"]
                    )
                    batches_per_epoch = num_samples // CONFIG["batch_size"]
                    # Repeat dataset for local epochs
                    dataset = dataset.repeat(CONFIG["epochs"])

                    trainer = LocalTrainer(
                        CONFIG["latent_dim"], CONFIG["learning_rate"], CONFIG["beta_1"]
                    )
                    # Initialize local trainer with current global weights
                    trainer.generator.set_weights(global_generator.get_weights())
                    trainer.discriminator.set_weights(
                        global_discriminator.get_weights()
                    )

                    # Accumulators for losses per epoch
                    epoch_gen_loss_sum = 0.0
                    epoch_disc_loss_sum = 0.0
                    epoch_batch_count = 0
                    client_gen_losses_this_round = []  # list for each epoch
                    client_disc_losses_this_round = []

                    for batch_idx, batch_images in enumerate(dataset):
                        gen_loss, disc_loss = trainer.train_step(batch_images)
                        epoch_gen_loss_sum += gen_loss.numpy()
                        epoch_disc_loss_sum += disc_loss.numpy()
                        epoch_batch_count += 1

                        # Check if an epoch is complete
                        if (batch_idx + 1) % batches_per_epoch == 0:
                            avg_gen_loss = epoch_gen_loss_sum / epoch_batch_count
                            avg_disc_loss = epoch_disc_loss_sum / epoch_batch_count
                            client_gen_losses_this_round.append(avg_gen_loss)
                            client_disc_losses_this_round.append(avg_disc_loss)
                            # Reset accumulators
                            epoch_gen_loss_sum = 0.0
                            epoch_disc_loss_sum = 0.0
                            epoch_batch_count = 0
                            # Stop if reached desired number of epochs
                            if len(client_gen_losses_this_round) >= CONFIG["epochs"]:
                                break
                    # Append the losses for this round to the overall history for this client
                    loss_history[num_clusters][cluster_id]["gen"].extend(
                        client_gen_losses_this_round
                    )
                    loss_history[num_clusters][cluster_id]["disc"].extend(
                        client_disc_losses_this_round
                    )

                    local_g_weights.append(trainer.generator.get_weights())
                    local_d_weights.append(trainer.discriminator.get_weights())
                except Exception as e:
                    logger.error("Error in cluster %d: %s", cluster_id, str(e))
                    continue

            if local_g_weights:
                # Average local weights and update global models
                avg_g_weights = average_weights(local_g_weights)
                avg_d_weights = average_weights(local_d_weights)
                global_generator.set_weights(avg_g_weights)
                global_discriminator.set_weights(avg_d_weights)

                round_dir = os.path.join(
                    CONFIG["base_models_path"],
                    f"clusters_{num_clusters}",
                    f"round_{round_num + 1}",
                )
                os.makedirs(round_dir, exist_ok=True)
                global_generator.save(os.path.join(round_dir, "global_generator.keras"))
                global_discriminator.save(
                    os.path.join(round_dir, "global_discriminator.keras")
                )
                generate_and_save_images(
                    global_generator,
                    round_num + 1,
                    os.path.join(
                        CONFIG["generated_images_path"],
                        f"federated/clusters_{num_clusters}",
                    ),
                )
                # Optionally, you can call evaluate_generator here, though the realism score is less precise.
                avg_score = evaluate_generator(
                    global_generator, unfed_discriminator, num_clusters, round_num + 1
                )
                with round_writer.as_default():
                    tf.summary.scalar("global_model_score", avg_score, step=round_num)
            else:
                logger.warning(
                    "No local weights collected for %d clusters in round %d",
                    num_clusters,
                    round_num + 1,
                )

        # After all rounds for the current client setting, plot the loss curves.
        # The x-axis will have composite labels "R{round}E{epoch}"
        total_points = CONFIG["rounds"] * CONFIG["epochs"]
        x_ticks = list(range(1, total_points + 1))
        x_labels = []
        for r in range(1, CONFIG["rounds"] + 1):
            for e in range(1, CONFIG["epochs"] + 1):
                x_labels.append(f"R{r}E{e}")

        # Plot Generator Loss for all clients in this setting
        plt.figure(figsize=(10, 6))
        for cluster_id, losses in loss_history[num_clusters].items():
            pts = range(1, len(losses["gen"]) + 1)
            plt.plot(pts, losses["gen"], marker="o", label=f"Client {cluster_id}")
        plt.xlabel("Round and Epoch")
        plt.ylabel("Generator Loss")
        plt.title(f"Generator Loss Curve for {num_clusters}-Client Setting")
        plt.xticks(x_ticks, x_labels, rotation=45, fontsize=8)
        plt.legend()
        gen_plot_path = os.path.join(
            CONFIG["eval_results_path"], f"loss_generator_{num_clusters}_clients.png"
        )
        plt.tight_layout()
        plt.savefig(gen_plot_path)
        plt.close()
        logger.info("Saved generator loss plot to: %s", gen_plot_path)

        # Plot Discriminator Loss for all clients in this setting
        plt.figure(figsize=(10, 6))
        for cluster_id, losses in loss_history[num_clusters].items():
            pts = range(1, len(losses["disc"]) + 1)
            plt.plot(pts, losses["disc"], marker="o", label=f"Client {cluster_id}")
        plt.xlabel("Round and Epoch")
        plt.ylabel("Discriminator Loss")
        plt.title(f"Discriminator Loss Curve for {num_clusters}-Client Setting")
        plt.xticks(x_ticks, x_labels, rotation=45, fontsize=8)
        plt.legend()
        disc_plot_path = os.path.join(
            CONFIG["eval_results_path"],
            f"loss_discriminator_{num_clusters}_clients.png",
        )
        plt.tight_layout()
        plt.savefig(disc_plot_path)
        plt.close()
        logger.info("Saved discriminator loss plot to: %s", disc_plot_path)


# Main Execution
if __name__ == "__main__":
    verify_paths()
    # First, train the unfederated model (now with loss tracking and plotting)
    unfed_generator, unfed_discriminator = train_unfederated_dcgan(
        CONFIG["unfederated_data_path"]
    )
    # Then run federated training using the unfederated discriminator as reference
    if unfed_discriminator:
        train_federated_dcgan(unfed_discriminator)
    else:
        logger.error(
            "Failed to start federated training: no unfederated discriminator available"
        )
