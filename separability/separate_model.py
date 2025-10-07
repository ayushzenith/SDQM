import os

import autokeras as ak
import keras  # if there is an error, use import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tempfile


def separate(synthetic_embeddings, real_embeddings, output_dir=None, max_params=None):
    # if output_dir is None, create a temporary directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    os.makedirs(os.path.join(output_dir, "autokeras"), exist_ok=True)

    # Set current working directory to the output directory
    os.chdir(os.path.join(output_dir, "autokeras"))

    # To numpy arrays
    synthetic_embeddings = np.array(synthetic_embeddings)
    real_embeddings = np.array(real_embeddings)

    # Create labels the same length as the embeddings
    synthetic_labels = np.array([0] * len(synthetic_embeddings))
    real_labels = np.array([1] * len(real_embeddings))

    # Combine the synthetic and real embeddings and labels
    data = np.concatenate((synthetic_embeddings, real_embeddings))
    labels = np.concatenate((synthetic_labels, real_labels))

    # shuffle the data
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.15, random_state=42
    )

    # Create the input node
    input_node = ak.Input()

    # Specify metrics in the ClassificationHead
    output_node = ak.ClassificationHead(metrics=["accuracy"])(input_node)

    # Initialize the AutoModel
    clf = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        overwrite=True,
        max_model_size=max_params,
    )

    # Train the model with validation_data
    history = clf.fit(
        x_train, y_train, epochs=1, batch_size=2, validation_data=(x_val, y_val)
    )

    # After training the model
    model = clf.export_model()

    # Print the model summary
    model.summary()


    # Get best validation accuracy
    best_accuracy = max(history.history["val_accuracy"])
    print(f"Best validation accuracy: {best_accuracy}")

    num_params = model.count_params()
    print(f"Best model number of parameters: {num_params}")

    # Save the model summary to a file
    with open('model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f"\nValidation Accuracy: {best_accuracy}\n")
        f.write(f"Number of Parameters: {num_params}\n")

    return num_params, best_accuracy
