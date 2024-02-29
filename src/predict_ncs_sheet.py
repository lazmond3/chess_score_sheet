from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
import tensorflow.keras.layers.experimental.preprocessing

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import argparse

# TODO introduce logger

# Split Dataset into 1) train-90% 2) validation-5% 3) test-5%
train_ratio = {
        "train": 0.9,
        "validation": 0.05,
        "test": 0.05,
        }
assert sum(train_ratio.values()) == 1


# TODO Add argparse options to change default values
class ModelParams():
    def __init__(self):
        self.batch_size = 64
        self.padding_token = 99
        self.image_width = 256
        self.image_height = 64
global model_params
model_params = ModelParams()


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def get_dataset_ncs(args):
    """
    Load NCS dataset
    Args:
        args: input option with argparser
    """
    image_path_list = os.listdir(args.input_pred)
    image_path_list = [args.input_pred + '/' + path for path in image_path_list]

    ret = {
        "all": {"img_paths": image_path_list,
                "labels": ['dummy' for i in range(len(image_path_list))]},
           }

    return ret


def get_dataset(args):
    """
    Load IAM dataset
    Args:
        args: input option with argparser
    """
    words_list = []

    words = open(f"{args.input}/train_val_data.txt", "r").readlines()
    for line in words:
        if line[0] == "#":
            continue
        words_list.append(line)

    np.random.shuffle(words_list)

    split_idx = int(train_ratio["train"] * len(words_list))
    train_samples = words_list[:split_idx]
    test_samples = words_list[split_idx:]

    validation_ratio = train_ratio["validation"] / (1 - train_ratio["train"])
    val_split_idx = int(validation_ratio * len(test_samples))
    validation_samples = test_samples[:val_split_idx]
    test_samples = test_samples[val_split_idx:]

    # Print splitting results
    print(f"Number of words_list: {len(words_list)}")
    print(f"Total training samples: {len(train_samples)}")
    print(f"Total validation samples: {len(validation_samples)}")
    print(f"Total test samples: {len(test_samples)}")

    # Create the list of image path and labels (w/ all the info)

    base_image_path = args.input

    def get_image_paths_and_labels(samples):
        paths = []
        corrected_samples = []
        for (i, file_line) in enumerate(samples):
            line_split = file_line.strip()
            line_split = line_split.split(" ")

            image_path = line_split[0]
            img_path = os.path.join(
                base_image_path, image_path
            )
            if os.path.getsize(img_path):
                paths.append(img_path)
                corrected_samples.append(file_line.split("\n")[0])

        return paths, corrected_samples

    train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
    validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
    test_img_paths, test_labels = get_image_paths_and_labels(test_samples)

    # Create the list of `clean` labels (w/ only the transcription part of the label )
    def clean_labels(labels):
        """
        `clean` means only the transcription part of the label
        """
        cleaned_labels = []
        for label in labels:
            label = label.split(" ")[-1].strip()
            cleaned_labels.append(label)
        return cleaned_labels

    train_labels_cleaned = clean_labels(train_labels)
    validation_labels_cleaned = clean_labels(validation_labels)
    test_labels_cleaned = clean_labels(test_labels)

    ret = {
        "train":      {"img_paths": train_img_paths,
                       "labels": train_labels_cleaned},
        "validation": {"img_paths": validation_img_paths,
                       "labels": validation_labels_cleaned},
        "test":       {"img_paths": test_img_paths,
                       "labels": test_labels_cleaned},
           }

    return ret


def distortion_free_resize(image):
    w, h = model_params.image_width, model_params.image_height
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]],
                   constant_values=model_params.padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(model_params.batch_size).cache().prefetch(tf.data.AUTOTUNE)


def replace_intermediate_layer_in_keras(model, layer_id, new_layer):
    layers = [layer for layer in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = keras.models.Model(input=layers[0].input, output=x)
    return new_model


def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)

    new_model = keras.models.Model(input=layers[0].input, output=x)
    return new_model


def main(args):
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        tf.random.set_seed(args.random_seed)

    ######################
    # Dataset preprocess #
    ######################
    dataset_map = get_dataset(args)
    dataset_map_ncs = get_dataset_ncs(args)

    # Build the character vocabulary
    global max_len  # FIXME global variable
    max_len = 0
    characters = set()

    for label in dataset_map["train"]["labels"]:
        max_len = max(max_len, len(label))
        for char in label:
            characters.add(char)

    characters = sorted(list(characters))

    print("Maximum length: ", max_len)
    print("Vocab size: ", len(characters))

    # Mapping characters to integers.
    global char_to_num
    char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

    # Mapping integers back to original characters.
    global num_to_char
    num_to_char = StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    train_ds = prepare_dataset(dataset_map["train"]["img_paths"],
                               dataset_map["train"]["labels"])
    validation_ds = prepare_dataset(dataset_map["validation"]["img_paths"],
                                    dataset_map["validation"]["labels"])
    test_ds = prepare_dataset(dataset_map["test"]["img_paths"],
                              dataset_map["test"]["labels"])

    # print dataset samples
    for data in train_ds.take(1):
        images, labels = data["image"], data["label"]

        _, ax = plt.subplots(4, 4, figsize=(15, 8))

        for i in range(16):
            img = images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            # Gather indices where label!= padding_token.
            label = labels[i]
            indices = tf.gather(label, tf.where(
                tf.math.not_equal(label, model_params.padding_token)))
            # Convert to string.
            label = tf.strings.reduce_join(num_to_char(indices))
            label = label.numpy().decode("utf-8")

            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis("off")

    if args.plt_save:
        plt.savefig('input_samples.png')

    ################
    # Define model #
    ################
    model = keras.saving.load_model(args.pretrained_model)

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    prediction_model.summary()

    ########
    # Test #
    ########

    print("Start testing")

    # A utility function to decode the output of the network.
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search.
        ctc_decode_ret = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
        results = ctc_decode_ret[0][0][:, :max_len]
        results_confidence = ctc_decode_ret[1]
        # Iterate over the results and get back the text.
        output_text = []
        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text, results_confidence

    #  Let's check results on some test samples.
    img_paths = dataset_map_ncs["all"]["img_paths"]
    labels = dataset_map_ncs["all"]["labels"]
    test_ds = prepare_dataset(img_paths, labels)

    f = open(args.predict_output, 'w')
    f.write("img_path,prediction_unconfidence,predicted_text")
    for batch_idx, batch in enumerate(test_ds):
        batch_images = batch["image"]
        _, ax = plt.subplots(4, 4, figsize=(15, 8))

        preds = prediction_model.predict(batch_images)
        pred_texts, pred_confidence = decode_batch_predictions(preds)

        # FIXME: hardcoded batch_size=64
        for i in range(64):
            if i >= len(preds):
                break

            img_path_idx = 64 * batch_idx + i
            f.write(f"{img_paths[img_path_idx]},{pred_confidence[i][0]},{pred_texts[i]}\n")


        for i in range(16):
            if i >= len(preds):
                break

            img = batch_images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            title = f"Pred: {pred_texts[i]} (unconf: {pred_confidence[i][0]:.3})"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")

        if args.plt_save:
            plt.savefig(f"output/prediction_samples_{batch_idx}.png")
    f.close()

    print("End testing")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default="data/HCS_Dataset_December_2021/extracted_move_boxes/")
    parser.add_argument('--input_pred', default="data/kanagawa_champ_2023/images/R1_B1_player1_sheet1.png_moves")
    parser.add_argument('--pretrained_model', default="output_model_hcs/")
    parser.add_argument('--predict_output', default="output/ncs_predict_result.txt")
    parser.add_argument('--random_seed', '-r', type=int, default=None)
    parser.add_argument('--plt_save', '-p', action='store_true')
    args = parser.parse_args()

    main(args)
