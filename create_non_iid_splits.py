import tensorflow as tf
import os
import random

client_settings = [3, 5, 7, 10]
data_path = "data/diabetic-retinopath-128-16-labeled.tfrecord"
split_tfrecord_path = "data/train-val-test"


def parse_record(example_proto):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    return parsed["image"], parsed["label"]


def load_dataset(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP")
    return list(
        dataset.map(
            parse_record, num_parallel_calls=tf.data.AUTOTUNE
        ).as_numpy_iterator()
    )


def write_tfrecords(data_records, tfrecord_path):
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(tfrecord_path, options) as writer:
        for img, lbl in data_records:
            feature = {
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[lbl])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def create_non_iid_splits(data_list, num_clients, min_samples_percent=0.01):
    label_groups = {}
    # Group samples by label
    for img, lbl in data_list:
        label_groups.setdefault(lbl, []).append((img, lbl))

    client_splits = [[] for _ in range(num_clients)]
    all_labels = list(label_groups.keys())
    random.shuffle(all_labels)

    # Initial non-IID distribution
    for label in all_labels:
        chosen_clients = random.sample(range(num_clients), k=min(2, num_clients))
        for img, lbl in label_groups[label]:
            client_splits[random.choice(chosen_clients)].append((img, lbl))

    # Calculate minimum samples per client (at least 1% of total data)
    total_samples = len(data_list)
    min_samples = max(1, int(total_samples * min_samples_percent))
    print(
        f"\nMinimum samples per client: {min_samples} ({(min_samples_percent*100):.1f}% of total)"
    )

    # Redistribution phase
    for client_id in range(num_clients):
        while len(client_splits[client_id]) < min_samples:
            # Find client with most samples
            richest_client = max(
                range(num_clients), key=lambda x: len(client_splits[x])
            )

            if len(client_splits[richest_client]) <= min_samples:
                print(
                    f"⚠️ Cannot fulfill minimum for client {client_id+1}. Maximum redistribution achieved."
                )
                break

            if len(client_splits[richest_client]) == 0:
                print(
                    f"🚨 Critical: No samples available for redistribution to client {client_id+1}"
                )
                break

            # Transfer sample from richest to needy client
            transferred_sample = client_splits[richest_client].pop()
            client_splits[client_id].append(transferred_sample)

    return client_splits


def split_dataset_non_iid(tfrecord_path, num_clients, output_dir):
    cluster_dir = os.path.join(output_dir, f"non_iid_clusters_{num_clients}")
    os.makedirs(cluster_dir, exist_ok=True)

    data_list = load_dataset(tfrecord_path)
    random.shuffle(data_list)
    client_splits = create_non_iid_splits(data_list, num_clients)

    # Write splits and metadata
    metadata_path = os.path.join(cluster_dir, "split_metadata.txt")
    with open(metadata_path, "w") as f:
        total_samples = sum(len(split) for split in client_splits)
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Client count: {num_clients}\n\n")

        for i, split in enumerate(client_splits):
            split_path = os.path.join(cluster_dir, f"non_iid_split_{i+1}.tfrecord")
            write_tfrecords(split, split_path)
            f.write(f"split_{i+1}: {len(split)} samples\n")
            print(f"Created {split_path} with {len(split)} samples")


if __name__ == "__main__":
    for client in client_settings:
        split_dataset_non_iid(data_path, client, split_tfrecord_path)
        print(f"\nCreated {client} non-IID clusters with minimum sample guarantee\n")
