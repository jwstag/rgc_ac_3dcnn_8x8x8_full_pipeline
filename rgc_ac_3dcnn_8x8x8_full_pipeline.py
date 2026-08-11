"""
Summer 2026 - Jiang Lab
Author: Jessica Wang

PURPOSE
-------
This script implements the complete pipeline for classifying retinal
ganglion cells (RGCs) and displaced amacrine cells (ACs) from HD-MEA
electrical images using a 3D convolutional neural network.

Pipeline:
    Raw HD-MEA .h5 files
    -> metadata extraction
    -> RGC/AC filtering and binary labeling
    -> 50 × 65 × 65 electrical-image extraction
    -> normalization
    -> 70/15/15 stratified split
    -> 8 × 8 × 8 3D CNN training
    -> evaluation and model saving

Labels:
    0 = AC
    1 = RGC
    """


# =============================================================================
# 1. IMPORTS
# =============================================================================

import math
import os
import pickle
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# =============================================================================
# 2. USER SETTINGS
# =============================================================================

# ---------------------------------------------------------------------------
# Raw data location
# ---------------------------------------------------------------------------

# Folder containing the original HD-MEA .h5 recordings.
RAW_DATA_DIR = Path(r"INSERT DATA PATH HERE")

# ---------------------------------------------------------------------------
# Preprocessing behavior
# ---------------------------------------------------------------------------

# True:
#   Rebuild metadata + STA arrays from the raw .h5 files.
#
# False:
#   Skip raw preprocessing and use existing processed files.
RUN_PREPROCESSING = False

# Existing/generated processed-data locations.
METADATA_PATH = Path("labeled_metadata.csv")
PROCESSED_DIR = Path("processed")
RAW_STA_PATH = PROCESSED_DIR / "sta_images.npy"
NORMALIZED_STA_PATH = PROCESSED_DIR / "sta_images_normalized.npy"

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

SEED = 42

# IMPORTANT:
# This is the batch size used in the controlled 8 x 8 x 8 kernel experiment.
BATCH_SIZE = 8

EPOCHS = 10
LEARNING_RATE = 0.0005
DROPOUT = 0.4

# Forced binary classification threshold.
BINARY_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Human-in-the-loop settings
# ---------------------------------------------------------------------------

RUN_UNCERTAINTY_ANALYSIS = True

# Threshold search is performed on validation data only.
# At least this proportion of validation cells must be automatically classified.
MIN_VALIDATION_COVERAGE = 0.80

# ---------------------------------------------------------------------------
# Output folder
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("8x8x8_model_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3. REPRODUCIBILITY
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
print("Output folder:", OUTPUT_DIR.resolve())


# =============================================================================
# 4. RAW HDF5 PREPROCESSING
# =============================================================================

def collect_metadata(raw_data_dir):
    """
    Read every .h5 file in raw_data_dir and build one metadata table.

    For each unit, save:
        file
        unit
        cell_type
        row
        column

    """

    h5_files = sorted(raw_data_dir.glob("*.h5"))

    if not h5_files:
        raise FileNotFoundError(
            f"No .h5 files were found in:\n{raw_data_dir.resolve()}"
        )

    print(f"Found {len(h5_files)} HDF5 files.")

    records = []

    for file_number, filename in enumerate(h5_files, start=1):

        print(
            f"Reading metadata from file "
            f"{file_number}/{len(h5_files)}: {filename.name}"
        )

        with h5py.File(filename, "r") as f:

            if "units" not in f:
                raise KeyError(
                    f"'units' group not found in {filename}"
                )

            for unit_name in f["units"].keys():

                unit = f["units"][unit_name]

                cell_type_raw = unit["cell_type"][()]

                if isinstance(cell_type_raw, bytes):
                    cell_type = cell_type_raw.decode()
                else:
                    cell_type = str(cell_type_raw)

                records.append({
                    "file": str(filename),
                    "unit": unit_name,
                    "cell_type": cell_type,
                    "row": int(unit["row"][()]),
                    "column": int(unit["column"][()]),
                })

    metadata = pd.DataFrame(records)

    print("\nAll cell types found:")
    print(metadata["cell_type"].value_counts(dropna=False))

    return metadata


def make_binary_metadata(metadata):
    """
    Keep only cells already labeled as:
        rgc
        ac

    Convert them to:
        rgc -> 1
        ac  -> 0

    Unknown cells are intentionally excluded from THIS binary model.
    They can be used later for the planned three-class model.
    """

    labeled = metadata[
        metadata["cell_type"].isin(["rgc", "ac"])
    ].copy()

    labeled["label"] = labeled["cell_type"].map({
        "rgc": 1,
        "ac": 0,
    })

    labeled.to_csv(
        METADATA_PATH,
        index=False,
    )

    print("\nBinary labeled dataset:")
    print(labeled["cell_type"].value_counts())
    print("Total labeled cells:", len(labeled))
    print("Saved:", METADATA_PATH.resolve())

    return labeled


def extract_sta_images(labeled_metadata):
    """
    Extract each cell's electrical image from:

        f["units"][unit]["eimage_sta"]["data"]

    Expected shape for every cell:
        50 x 65 x 65

    The full dataset is saved as a memory-mapped float32 .npy file so that
    the entire dataset does not need to remain in RAM.
    """

    expected_shape = (50, 65, 65)

    sta_file = np.lib.format.open_memmap(
        RAW_STA_PATH,
        mode="w+",
        dtype="float32",
        shape=(len(labeled_metadata), *expected_shape),
    )

    try:
        for i, (_, row) in enumerate(labeled_metadata.iterrows()):

            with h5py.File(row["file"], "r") as f:

                sta = f[
                    "units"
                ][
                    row["unit"]
                ][
                    "eimage_sta"
                ][
                    "data"
                ][()]

            if sta.shape != expected_shape:
                raise ValueError(
                    f"Unexpected STA shape for "
                    f"{row['file']} / {row['unit']}:\n"
                    f"Expected {expected_shape}, found {sta.shape}"
                )

            sta_file[i] = sta.astype(np.float32)

            if i % 1000 == 0:
                sta_file.flush()
                print(
                    f"Extracted {i}/{len(labeled_metadata)} cells"
                )

        sta_file.flush()

    finally:
        del sta_file

    print("STA extraction complete.")
    print("Saved:", RAW_STA_PATH.resolve())


def normalize_sta_images():
    """
    Normalize EACH cell independently using:

        normalized = (image - image_mean) / image_std

    If a cell has standard deviation = 0, its values are left unchanged.
    """

    sta = np.load(
        RAW_STA_PATH,
        mmap_mode="r",
    )

    normalized = np.lib.format.open_memmap(
        NORMALIZED_STA_PATH,
        mode="w+",
        dtype=np.float32,
        shape=sta.shape,
    )

    try:
        for i in range(sta.shape[0]):

            image = sta[i].astype(np.float32)

            mean = image.mean()
            std = image.std()

            if std > 0:
                normalized[i] = (
                    image - mean
                ) / std
            else:
                normalized[i] = image

            if i % 1000 == 0:
                normalized.flush()
                print(
                    f"Normalized {i}/{sta.shape[0]} cells"
                )

        normalized.flush()

    finally:
        del normalized

    print("Normalization complete.")
    print("Saved:", NORMALIZED_STA_PATH.resolve())


def validate_normalized_file():
    """
    Perform simple checks on the normalized dataset.
    """

    X_check = np.load(
        NORMALIZED_STA_PATH,
        mmap_mode="r",
    )

    print("\nNormalized array shape:", X_check.shape)
    print("Normalized dtype:", X_check.dtype)

    indices_to_check = [
        i for i in [0, 1, 100, 1000]
        if i < len(X_check)
    ]

    for i in indices_to_check:
        print(
            f"Cell {i}: "
            f"min={X_check[i].min():.4f}, "
            f"max={X_check[i].max():.4f}, "
            f"mean={X_check[i].mean():.4f}, "
            f"std={X_check[i].std():.4f}"
        )


def run_preprocessing():
    """
    Run the complete raw-data preprocessing pipeline.
    """

    print("\n" + "=" * 70)
    print("RUNNING RAW HDF5 PREPROCESSING")
    print("=" * 70)

    metadata = collect_metadata(
        RAW_DATA_DIR
    )

    # Save metadata for ALL cells, including unknowns.
    metadata.to_csv(
        "rgc_metadata.csv",
        index=False,
    )

    print(
        "Saved complete metadata:",
        Path("rgc_metadata.csv").resolve(),
    )

    labeled_metadata = make_binary_metadata(
        metadata
    )

    extract_sta_images(
        labeled_metadata
    )

    normalize_sta_images()

    validate_normalized_file()

    print("\nPreprocessing complete.")


# =============================================================================
# 5. RUN OR SKIP PREPROCESSING
# =============================================================================

if RUN_PREPROCESSING:

    run_preprocessing()

else:

    print("\nRUN_PREPROCESSING = False")
    print("Using existing processed files.")

    missing_files = [
        path
        for path in [
            METADATA_PATH,
            NORMALIZED_STA_PATH,
        ]
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Preprocessing was skipped, but these required files "
            "could not be found:\n"
            + "\n".join(str(path) for path in missing_files)
            + "\n\nEither set RUN_PREPROCESSING = True "
              "or place the existing processed files in these locations."
        )


# =============================================================================
# 6. LOAD THE NORMALIZED BINARY DATASET
# =============================================================================

X = np.load(
    NORMALIZED_STA_PATH,
    mmap_mode="r",
)

metadata = pd.read_csv(
    METADATA_PATH
)

if "label" not in metadata.columns:
    raise KeyError(
        "labeled_metadata.csv must contain a column named 'label'."
    )

y = metadata["label"].to_numpy(
    dtype=np.int32
)

print("\nLoaded training data:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print(
    "Class counts:",
    dict(
        zip(
            *np.unique(
                y,
                return_counts=True,
            )
        )
    ),
)


# =============================================================================
# 7. VALIDATE MODEL INPUT
# =============================================================================

assert len(X) == len(y), (
    "The number of electrical images and labels does not match."
)

assert X.ndim == 4, (
    f"Expected X shape (N, 50, 65, 65), but found {X.shape}."
)

assert X.shape[1:] == (50, 65, 65), (
    f"Expected each electrical image to be 50 x 65 x 65, "
    f"but found {X.shape[1:]}."
)

assert set(np.unique(y)).issubset({0, 1}), (
    "This model expects binary labels only: 0 = AC, 1 = RGC."
)

print("Model-input validation passed.")


# =============================================================================
# 8. 70 / 15 / 15 STRATIFIED SPLIT
# =============================================================================

all_indices = np.arange(
    len(y)
)

# 70% train, 30% temporary.
train_idx, temporary_idx = train_test_split(
    all_indices,
    test_size=0.30,
    random_state=SEED,
    stratify=y,
)

# Split the temporary 30% into:
# 15% validation
# 15% test
val_idx, test_idx = train_test_split(
    temporary_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=y[temporary_idx],
)

print("\nDataset split:")
print("Training:", len(train_idx))
print("Validation:", len(val_idx))
print("Testing:", len(test_idx))


# =============================================================================
# 9. TF.DATA PIPELINE
# =============================================================================

INPUT_SHAPE = (
    50,
    65,
    65,
    1,
)


def data_generator(indices):
    """
    Yield one image and one label at a time.

    Original stored shape:
        50 x 65 x 65

    CNN input shape:
        50 x 65 x 65 x 1
    """

    for index in indices:

        image = np.asarray(
            X[index],
            dtype=np.float32,
        )

        image = image[
            ...,
            np.newaxis
        ]

        label = np.float32(
            y[index]
        )

        yield image, label


def create_dataset(
    indices,
    training=False,
    repeat=False,
):
    """
    Build a TensorFlow Dataset.
    """

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(indices),
        output_signature=(
            tf.TensorSpec(
                shape=INPUT_SHAPE,
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(),
                dtype=tf.float32,
            ),
        ),
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(
                1000,
                len(indices),
            ),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_dataset = create_dataset(
    train_idx,
    training=True,
    repeat=True,
)

val_dataset = create_dataset(
    val_idx,
    training=False,
    repeat=True,
)

test_dataset = create_dataset(
    test_idx,
    training=False,
    repeat=False,
)


# =============================================================================
# 10. BUILD THE 8 x 8 x 8 MODEL
# =============================================================================

def build_8x8x8_model():
    """
    Build the strongest completed Summer 2026 binary CNN.

    Architecture:
        Input
        -> Conv3D(8 filters, 8x8x8)
        -> BatchNormalization
        -> MaxPooling3D
        -> Conv3D(16 filters, 8x8x8)
        -> BatchNormalization
        -> MaxPooling3D
        -> GlobalAveragePooling3D
        -> Dense(32)
        -> Dropout(0.4)
        -> Dense(1, sigmoid)
    """

    model = models.Sequential([

        layers.Input(
            shape=INPUT_SHAPE
        ),

        # ---------------------------------------------------------------------
        # Convolution block 1
        # ---------------------------------------------------------------------

        layers.Conv3D(
            filters=8,
            kernel_size=(8, 8, 8),
            activation="relu",
            padding="same",
        ),

        layers.BatchNormalization(),

        layers.MaxPooling3D(
            pool_size=(2, 2, 2)
        ),

        # ---------------------------------------------------------------------
        # Convolution block 2
        # ---------------------------------------------------------------------

        layers.Conv3D(
            filters=16,
            kernel_size=(8, 8, 8),
            activation="relu",
            padding="same",
        ),

        layers.BatchNormalization(),

        layers.MaxPooling3D(
            pool_size=(2, 2, 2)
        ),

        # ---------------------------------------------------------------------
        # Final classification layers
        # ---------------------------------------------------------------------

        layers.GlobalAveragePooling3D(),

        layers.Dense(
            32,
            activation="relu",
        ),

        layers.Dropout(
            DROPOUT
        ),

        layers.Dense(
            1,
            activation="sigmoid",
        ),
    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
        ],
    )

    return model


model = build_8x8x8_model()

print("\nModel architecture:")
model.summary()


# =============================================================================
# 11. TRAIN THE MODEL
# =============================================================================

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best_model_kernel_8x8x8.keras"
)

FINAL_MODEL_PATH = (
    OUTPUT_DIR
    / "final_model_kernel_8x8x8.keras"
)

checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1,
)

early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

steps_per_epoch = math.ceil(
    len(train_idx)
    / BATCH_SIZE
)

validation_steps = math.ceil(
    len(val_idx)
    / BATCH_SIZE
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[
        checkpoint,
        early_stop,
    ],
)

model.save(
    FINAL_MODEL_PATH
)

with open(
    OUTPUT_DIR / "training_history.pkl",
    "wb",
) as file:
    pickle.dump(
        history.history,
        file,
    )

print("\nTraining complete.")
print("Best checkpoint:", BEST_MODEL_PATH.resolve())
print("Final model:", FINAL_MODEL_PATH.resolve())


# =============================================================================
# 12. LOAD THE BEST CHECKPOINT
# =============================================================================

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)

print("Best validation-AUC checkpoint reloaded.")


# =============================================================================
# 13. CREATE VALIDATION AND TEST PROBABILITIES
# =============================================================================

val_eval_dataset = create_dataset(
    val_idx,
    training=False,
    repeat=False,
)

test_eval_dataset = create_dataset(
    test_idx,
    training=False,
    repeat=False,
)

val_probabilities = best_model.predict(
    val_eval_dataset,
    verbose=1,
).reshape(-1)

test_probabilities = best_model.predict(
    test_eval_dataset,
    verbose=1,
).reshape(-1)

y_val = y[
    val_idx
].astype(np.int32)

y_test = y[
    test_idx
].astype(np.int32)

np.save(
    OUTPUT_DIR / "validation_probabilities.npy",
    val_probabilities,
)

np.save(
    OUTPUT_DIR / "test_probabilities.npy",
    test_probabilities,
)


# =============================================================================
# 14. FORCED BINARY EVALUATION
# =============================================================================

binary_predictions = (
    test_probabilities
    >= BINARY_THRESHOLD
).astype(np.int32)

binary_metrics = {

    "threshold":
        BINARY_THRESHOLD,

    "accuracy":
        accuracy_score(
            y_test,
            binary_predictions,
        ),

    "balanced_accuracy":
        balanced_accuracy_score(
            y_test,
            binary_predictions,
        ),

    "auc":
        roc_auc_score(
            y_test,
            test_probabilities,
        ),

    "precision":
        precision_score(
            y_test,
            binary_predictions,
            zero_division=0,
        ),

    "recall":
        recall_score(
            y_test,
            binary_predictions,
            zero_division=0,
        ),

    "f1":
        f1_score(
            y_test,
            binary_predictions,
            zero_division=0,
        ),
}

print("\nFORCED BINARY TEST RESULTS")
print("=" * 45)

for name, value in binary_metrics.items():
    print(
        f"{name}: {value:.4f}"
    )

pd.DataFrame(
    [binary_metrics]
).to_csv(
    OUTPUT_DIR
    / "binary_test_metrics.csv",
    index=False,
)

binary_report = classification_report(
    y_test,
    binary_predictions,
    target_names=[
        "AC",
        "RGC",
    ],
    digits=4,
    zero_division=0,
)

print("\nClassification report:")
print(binary_report)

with open(
    OUTPUT_DIR
    / "binary_classification_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(
        binary_report
    )


# =============================================================================
# 15. CONFUSION MATRIX
# =============================================================================

binary_cm = confusion_matrix(
    y_test,
    binary_predictions,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=binary_cm,
    display_labels=[
        "AC",
        "RGC",
    ],
)

display.plot(
    values_format="d"
)

plt.title(
    "8 x 8 x 8 Test Confusion Matrix\n"
    "Threshold = 0.50"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "binary_confusion_matrix.png",
    dpi=300,
)

plt.close()


# =============================================================================
# 16. ROC CURVE
# =============================================================================

fpr, tpr, _ = roc_curve(
    y_test,
    test_probabilities,
)

test_auc = roc_auc_score(
    y_test,
    test_probabilities,
)

plt.figure(
    figsize=(6, 6)
)

plt.plot(
    fpr,
    tpr,
    label=(
        f"8 x 8 x 8 model, "
        f"AUC = {test_auc:.4f}"
    ),
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier",
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "8 x 8 x 8 Test ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "test_roc_curve.png",
    dpi=300,
)

plt.close()


# =============================================================================
# 17. SAVE INDIVIDUAL TEST PREDICTIONS
# =============================================================================

test_results = pd.DataFrame({

    "original_index":
        test_idx,

    "true_label":
        y_test,

    "true_class":
        np.where(
            y_test == 1,
            "RGC",
            "AC",
        ),

    "probability_RGC":
        test_probabilities,

    "binary_prediction":
        binary_predictions,

    "binary_prediction_class":
        np.where(
            binary_predictions == 1,
            "RGC",
            "AC",
        ),
})

test_results.to_csv(
    OUTPUT_DIR
    / "all_binary_test_predictions.csv",
    index=False,
)


# =============================================================================
# 18. OPTIONAL HUMAN-IN-THE-LOOP UNCERTAINTY ANALYSIS
# =============================================================================

def make_uncertainty_predictions(
    probabilities,
    lower,
    upper,
):
    """
    Return:
        0 = AC
        1 = RGC
        2 = Uncertain

    probability < lower:
        AC

    probability > upper:
        RGC

    otherwise:
        Uncertain / human review
    """

    predictions = np.full(
        len(probabilities),
        2,
        dtype=np.int32,
    )

    predictions[
        probabilities < lower
    ] = 0

    predictions[
        probabilities > upper
    ] = 1

    return predictions


if RUN_UNCERTAINTY_ANALYSIS:

    # -------------------------------------------------------------------------
    # 18A. FIND UNCERTAINTY THRESHOLDS USING VALIDATION DATA ONLY
    # -------------------------------------------------------------------------

    search_rows = []

    lower_thresholds = np.arange(
        0.05,
        0.50,
        0.01,
    )

    upper_thresholds = np.arange(
        0.51,
        0.96,
        0.01,
    )

    for lower in lower_thresholds:

        for upper in upper_thresholds:

            predictions = make_uncertainty_predictions(
                val_probabilities,
                lower,
                upper,
            )

            accepted_mask = (
                predictions != 2
            )

            coverage = (
                accepted_mask.mean()
            )

            if accepted_mask.sum() == 0:
                continue

            accepted_true = y_val[
                accepted_mask
            ]

            accepted_predictions = predictions[
                accepted_mask
            ]

            search_rows.append({

                "lower_threshold":
                    float(lower),

                "upper_threshold":
                    float(upper),

                "coverage":
                    float(coverage),

                "uncertain_rate":
                    float(
                        1.0
                        - coverage
                    ),

                "selective_accuracy":
                    accuracy_score(
                        accepted_true,
                        accepted_predictions,
                    ),

                "selective_balanced_accuracy":
                    balanced_accuracy_score(
                        accepted_true,
                        accepted_predictions,
                    ),

                "automatically_classified":
                    int(
                        accepted_mask.sum()
                    ),

                "sent_for_review":
                    int(
                        (~accepted_mask).sum()
                    ),
            })

    uncertainty_search = pd.DataFrame(
        search_rows
    )

    eligible = uncertainty_search[
        uncertainty_search[
            "coverage"
        ] >= MIN_VALIDATION_COVERAGE
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "No uncertainty threshold pair reached "
            f"the minimum coverage of "
            f"{MIN_VALIDATION_COVERAGE:.0%}."
        )

    selected_row = eligible.sort_values(
        [
            "selective_balanced_accuracy",
            "selective_accuracy",
            "coverage",
        ],
        ascending=False,
    ).iloc[0]

    LOWER_THRESHOLD = float(
        selected_row[
            "lower_threshold"
        ]
    )

    UPPER_THRESHOLD = float(
        selected_row[
            "upper_threshold"
        ]
    )

    uncertainty_search.to_csv(
        OUTPUT_DIR
        / "validation_uncertainty_threshold_search.csv",
        index=False,
    )

    print("\nSELECTED UNCERTAINTY THRESHOLDS")
    print("=" * 45)

    print(
        f"Lower threshold: "
        f"{LOWER_THRESHOLD:.2f}"
    )

    print(
        f"Upper threshold: "
        f"{UPPER_THRESHOLD:.2f}"
    )

    # -------------------------------------------------------------------------
    # 18B. APPLY THE CHOSEN THRESHOLDS TO THE TEST SET
    # -------------------------------------------------------------------------

    uncertainty_predictions = make_uncertainty_predictions(
        test_probabilities,
        LOWER_THRESHOLD,
        UPPER_THRESHOLD,
    )

    accepted_mask = (
        uncertainty_predictions
        != 2
    )

    uncertain_mask = (
        uncertainty_predictions
        == 2
    )

    accepted_true = y_test[
        accepted_mask
    ]

    accepted_predictions = uncertainty_predictions[
        accepted_mask
    ]

    uncertainty_metrics = {

        "lower_threshold":
            LOWER_THRESHOLD,

        "upper_threshold":
            UPPER_THRESHOLD,

        "coverage":
            float(
                accepted_mask.mean()
            ),

        "uncertain_rate":
            float(
                uncertain_mask.mean()
            ),

        "selective_accuracy":
            accuracy_score(
                accepted_true,
                accepted_predictions,
            ),

        "selective_balanced_accuracy":
            balanced_accuracy_score(
                accepted_true,
                accepted_predictions,
            ),

        "automatically_classified":
            int(
                accepted_mask.sum()
            ),

        "sent_for_review":
            int(
                uncertain_mask.sum()
            ),

        "total_test_samples":
            int(
                len(y_test)
            ),
    }

    print("\nHUMAN-IN-THE-LOOP TEST RESULTS")
    print("=" * 45)

    for name, value in uncertainty_metrics.items():

        if isinstance(
            value,
            float,
        ):
            print(
                f"{name}: "
                f"{value:.4f}"
            )

        else:
            print(
                f"{name}: "
                f"{value}"
            )

    pd.DataFrame(
        [uncertainty_metrics]
    ).to_csv(
        OUTPUT_DIR
        / "uncertainty_test_metrics.csv",
        index=False,
    )

    test_results[
        "uncertainty_prediction"
    ] = uncertainty_predictions

    test_results[
        "uncertainty_prediction_class"
    ] = np.select(
        [
            uncertainty_predictions == 0,
            uncertainty_predictions == 1,
        ],
        [
            "AC",
            "RGC",
        ],
        default="Uncertain",
    )

    test_results.to_csv(
        OUTPUT_DIR
        / "all_test_predictions_with_uncertainty.csv",
        index=False,
    )


# =============================================================================
# 19. SAVE A HUMAN-READABLE SUMMARY
# =============================================================================

summary_lines = [

    "RGC vs AC 3D CNN - 8 x 8 x 8",
    "=" * 45,
    "",

    "Forced binary classification:",

    f"Threshold: "
    f"{BINARY_THRESHOLD:.2f}",

    f"Accuracy: "
    f"{binary_metrics['accuracy']:.4f}",

    f"Balanced accuracy: "
    f"{binary_metrics['balanced_accuracy']:.4f}",

    f"AUC: "
    f"{binary_metrics['auc']:.4f}",

    f"Precision: "
    f"{binary_metrics['precision']:.4f}",

    f"Recall: "
    f"{binary_metrics['recall']:.4f}",

    f"F1: "
    f"{binary_metrics['f1']:.4f}",
]

if RUN_UNCERTAINTY_ANALYSIS:

    summary_lines.extend([
        "",
        "Human-in-the-loop uncertainty system:",

        f"Lower threshold: "
        f"{LOWER_THRESHOLD:.2f}",

        f"Upper threshold: "
        f"{UPPER_THRESHOLD:.2f}",

        f"Coverage: "
        f"{uncertainty_metrics['coverage']:.4f}",

        f"Uncertain rate: "
        f"{uncertainty_metrics['uncertain_rate']:.4f}",

        f"Selective accuracy: "
        f"{uncertainty_metrics['selective_accuracy']:.4f}",

        (
            "Selective balanced accuracy: "
            f"{uncertainty_metrics['selective_balanced_accuracy']:.4f}"
        ),

        (
            "Automatically classified: "
            f"{uncertainty_metrics['automatically_classified']}"
        ),

        (
            "Sent for human review: "
            f"{uncertainty_metrics['sent_for_review']}"
        ),
    ])

summary = "\n".join(
    summary_lines
)

print(
    "\n" + summary
)

with open(
    OUTPUT_DIR
    / "final_summary.txt",
    "w",
    encoding="utf-8",
) as file:

    file.write(
        summary
    )


# =============================================================================
# 20. DONE
# =============================================================================

print("\nDONE")
print("All outputs were saved to:")
print(OUTPUT_DIR.resolve())
