import os
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.random import normal
import matplotlib.pyplot as plt
from model_dcgan import build_generator, build_discriminator


# Constants
BATCH_SIZE = 64
NOISE_DIM = 200
LEARNING_RATE = 0.0002
BETA_1 = 0.5
EPOCHS = 1  # Running for 1 epoch

# Create directories for saving models and images
os.makedirs('models', exist_ok=True)
os.makedirs('imgs', exist_ok=True)


# Function to parse the TFRecord file and get the image
def _parse_image_function(example_proto):
    image_tensor = tf.io.parse_tensor(example_proto, out_type=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=-1)
    return image_tensor

# Create dataset from TFRecord
def create_dataset_from_tfrecord(tfrecord_path, batch_size, buffer_size=10000):
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP")
    dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Loss functions
def discriminator_loss(real_output, fake_output):
    real_loss = binary_crossentropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return binary_crossentropy(tf.ones_like(fake_output), fake_output)

# Training step
@tf.function
def train_step(images, generator, discriminator, gen_optimizer, disc_optimizer, noise_dim):
    noise = normal([tf.shape(images)[0], noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    return gen_loss, disc_loss

# Save generated images
def save_images(generator, test_input, epoch):
    predictions = generator(test_input, training=False)
    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.savefig(f'imgs/image_at_epoch_{epoch + 1:04d}.png')
    plt.close()

# Main training function
def train(tfrecord_file_path, epochs, noise_dim, batch_size, learning_rate, beta_1):
    # Initialize new generator and discriminator
    print("Creating new Generator and Discriminator")
    generator = build_generator(noise_dim)
    discriminator = build_discriminator()

    # Optimizers
    generator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1)
    discriminator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1)

    # Prepare dataset
    dataset = create_dataset_from_tfrecord(tfrecord_file_path, batch_size)

    # Seed noise for consistent image saving
    seed_noise = normal([16, noise_dim])

    # Training loop
    step = 0
    for epoch in range(epochs):
        for image_batch in dataset:
            step += 1
            print(f"Step: {step}")
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, noise_dim)

        # Save model and generated images at the end of the epoch
        generator.save(f'models/generator_epoch_{epoch + 1:04d}.keras')
        discriminator.save(f'models/discriminator_epoch_{epoch + 1:04d}.keras')
        print(f"Epoch {epoch + 1}: Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
        save_images(generator, seed_noise, epoch)

if __name__ == '__main__':
    tfrecord_file_path = "./data/pretraining/rsna-abdominal-128-16.tfrecord"  # Specify your TFRecord file path
    train(tfrecord_file_path, epochs=EPOCHS, noise_dim=NOISE_DIM, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, beta_1=BETA_1)

