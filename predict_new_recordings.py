"""
Apply the trained 8 x 8 x 8 RGC-vs-AC 3D CNN to NEW HD-MEA recordings.

Summer 2026 - Jiang Lab
Author: Jessica Wang

PURPOSE
-------
This script loads a previously trained 8 x 8 x 8 3D CNN and applies it to
new HD-MEA .h5 recordings WITHOUT retraining the model.

Pipeline:
    New HD-MEA .h5 file(s)
    -> extract each unit's 50 x 65 x 65 electrical image
    -> normalize each electrical image using its own mean and standard deviation
    -> load trained 8 x 8 x 8 CNN
    -> predict probability of RGC
    -> assign AC / RGC / Uncertain
    -> save predictions to CSV

IMPORTANT
---------
This script does NOT add new cells to the training dataset and does NOT retrain
the model. It is only for inference/prediction on new recordings.

Expected HDF5 structure for each unit:
    units/<unit_name>/eimage_sta/data

Expected electrical-image shape:
    50 x 65 x 65
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf


# =============================================================================
# 2. USER SETTINGS
# =============================================================================

# ---------------------------------------------------------------------------
# New data location
# ---------------------------------------------------------------------------

# Option A:
# Put one or more new .h5 files inside a folder named "new_data".
NEW_DATA_DIR = Path("new_data")

# Option B:
# If you want to use only one specific .h5 file, enter its path here.
# Leave as None to process every .h5 file inside NEW_DATA_DIR.
SINGLE_H5_FILE = None
# Example:
# SINGLE_H5_FILE = Path("new_data/example_recording.h5")


# ---------------------------------------------------------------------------
# Trained model location
# ---------------------------------------------------------------------------

# Path to the saved 8 x 8 x 8 model from the training pipeline.
MODEL_PATH = Path(
    "8x8x8_model_outputs/best_model_kernel_8x8x8.keras"
)


# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------

# Standard forced-binary threshold:
# probability >= 0.50 -> RGC
# probability < 0.50  -> AC
BINARY_THRESHOLD = 0.50

# Human-in-the-loop thresholds from the selected uncertainty strategy:
#
# probability < 0.08  -> AC
# probability > 0.95  -> RGC
# otherwise           -> Uncertain / human review
LOWER_THRESHOLD = NONE #insert value; in the orignal 8x8x8 I used 0.08
UPPER_THRESHOLD = NONE #insert value; in the orignal 8x8x8 I used 0.95

# Turn this off if you only want forced binary AC/RGC predictions.
USE_UNCERTAINTY_CLASSIFICATION = False


# ---------------------------------------------------------------------------
# Prediction settings
# ---------------------------------------------------------------------------

BATCH_SIZE = 8

OUTPUT_DIR = Path("new_data_predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3. FIND NEW HDF5 FILES
# =============================================================================

def get_h5_files():
    """
    Return the new HDF5 files that should be classified.

    If SINGLE_H5_FILE is provided, only that file is used.
    Otherwise, every .h5 file inside NEW_DATA_DIR is used.
    """

    if SINGLE_H5_FILE is not None:

        file_path = Path(SINGLE_H5_FILE)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find the HDF5 file:\n{file_path.resolve()}"
            )

        if file_path.suffix.lower() != ".h5":
            raise ValueError(
                f"Expected a .h5 file, but received:\n{file_path}"
            )

        return [file_path]

    h5_files = sorted(
        NEW_DATA_DIR.glob("*.h5")
    )

    if not h5_files:
        raise FileNotFoundError(
            f"No .h5 files were found inside:\n"
            f"{NEW_DATA_DIR.resolve()}"
        )

    return h5_files


# =============================================================================
# 4. LOAD THE TRAINED MODEL
# =============================================================================

def load_trained_model():
    """
    Load the saved 8 x 8 x 8 Keras model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find the trained model:\n"
            f"{MODEL_PATH.resolve()}"
        )

    print("Loading model:")
    print(MODEL_PATH.resolve())

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    return model


# =============================================================================
# 5. NORMALIZE ONE ELECTRICAL IMAGE
# =============================================================================

def normalize_electrical_image(image):
    """
    Normalize one 50 x 65 x 65 electrical image using the same method
    as the training preprocessing pipeline:

        normalized = (image - mean) / standard_deviation

    If standard deviation is zero, the image is returned unchanged.
    """

    image = image.astype(
        np.float32
    )

    mean = image.mean()
    std = image.std()

    if std > 0:
        image = (
            image - mean
        ) / std

    return image


# =============================================================================
# 6. READ ONE UNIT FROM AN HDF5 FILE
# =============================================================================

def extract_unit_image(unit, file_name, unit_name):
    """
    Extract and normalize one unit's electrical image.

    Expected dataset:
        unit["eimage_sta"]["data"]

    Expected shape:
        50 x 65 x 65
    """

    expected_shape = (
        50,
        65,
        65,
    )

    if "eimage_sta" not in unit:
        raise KeyError(
            f"Missing 'eimage_sta' for unit "
            f"{unit_name} in {file_name}"
        )

    if "data" not in unit["eimage_sta"]:
        raise KeyError(
            f"Missing 'eimage_sta/data' for unit "
            f"{unit_name} in {file_name}"
        )

    image = unit[
        "eimage_sta"
    ][
        "data"
    ][()]

    if image.shape != expected_shape:
        raise ValueError(
            f"Unexpected electrical-image shape for "
            f"{file_name} / {unit_name}.\n"
            f"Expected {expected_shape}, found {image.shape}."
        )

    image = normalize_electrical_image(
        image
    )

    # Add one channel dimension:
    # 50 x 65 x 65 -> 50 x 65 x 65 x 1
    image = image[
        ...,
        np.newaxis
    ]

    return image


# =============================================================================
# 7. SAFELY READ OPTIONAL UNIT METADATA
# =============================================================================

def read_optional_scalar(unit, key):
    """
    Read a scalar metadata field if it exists.

    Returns None if the field is not available.
    """

    if key not in unit:
        return None

    value = unit[key][()]

    if isinstance(value, bytes):
        return value.decode()

    if isinstance(value, np.generic):
        return value.item()

    return value


# =============================================================================
# 8. CLASSIFICATION FUNCTIONS
# =============================================================================

def forced_binary_class(probability_rgc):
    """
    Convert RGC probability into a forced binary prediction.
    """

    if probability_rgc >= BINARY_THRESHOLD:
        return "RGC"

    return "AC"


def uncertainty_class(probability_rgc):
    """
    Apply the selected human-in-the-loop uncertainty thresholds.

    Below LOWER_THRESHOLD:
        AC

    Above UPPER_THRESHOLD:
        RGC

    Between thresholds:
        Uncertain
    """

    if probability_rgc < LOWER_THRESHOLD:
        return "AC"

    if probability_rgc > UPPER_THRESHOLD:
        return "RGC"

    return "Uncertain"


# =============================================================================
# 9. CLASSIFY ONE HDF5 RECORDING
# =============================================================================

def classify_h5_file(model, h5_path):
    """
    Classify all usable units in one HDF5 recording.

    Returns a DataFrame containing one row per successfully processed unit.
    """

    print("\n" + "=" * 70)
    print("Processing:")
    print(h5_path.resolve())
    print("=" * 70)

    images = []
    metadata_rows = []
    skipped_units = []

    with h5py.File(
        h5_path,
        "r",
    ) as f:

        if "units" not in f:
            raise KeyError(
                f"'units' group not found in:\n{h5_path}"
            )

        unit_names = list(
            f["units"].keys()
        )

        print(
            f"Units found: {len(unit_names)}"
        )

        for unit_name in unit_names:

            unit = f[
                "units"
            ][
                unit_name
            ]

            try:
                image = extract_unit_image(
                    unit=unit,
                    file_name=h5_path.name,
                    unit_name=unit_name,
                )

            except (
                KeyError,
                ValueError,
            ) as error:

                print(
                    f"Skipping unit {unit_name}: {error}"
                )

                skipped_units.append({
                    "recording": h5_path.name,
                    "unit": unit_name,
                    "reason": str(error),
                })

                continue

            images.append(
                image
            )

            metadata_rows.append({
                "recording": h5_path.name,
                "unit": unit_name,

                # These are included only if present in the file.
                # They are NOT used by the CNN for prediction.
                "existing_cell_type":
                    read_optional_scalar(
                        unit,
                        "cell_type",
                    ),

                "row":
                    read_optional_scalar(
                        unit,
                        "row",
                    ),

                "column":
                    read_optional_scalar(
                        unit,
                        "column",
                    ),
            })

    if not images:
        raise RuntimeError(
            f"No usable 50 x 65 x 65 electrical images "
            f"were found in {h5_path.name}."
        )

    # Convert the list into one CNN input array:
    # N x 50 x 65 x 65 x 1
    X_new = np.stack(
        images
    ).astype(
        np.float32
    )

    print(
        "Prediction input shape:",
        X_new.shape,
    )

    # -------------------------------------------------------------------------
    # Run the trained CNN
    # -------------------------------------------------------------------------

    probabilities = model.predict(
        X_new,
        batch_size=BATCH_SIZE,
        verbose=1,
    ).reshape(-1)

    # -------------------------------------------------------------------------
    # Build output table
    # -------------------------------------------------------------------------

    results = pd.DataFrame(
        metadata_rows
    )

    results[
        "probability_RGC"
    ] = probabilities

    results[
        "forced_binary_prediction"
    ] = [
        forced_binary_class(
            probability
        )
        for probability in probabilities
    ]

    if USE_UNCERTAINTY_CLASSIFICATION:

        results[
            "HITL_prediction"
        ] = [
            uncertainty_class(
                probability
            )
            for probability in probabilities
        ]

        results[
            "needs_human_review"
        ] = (
            results[
                "HITL_prediction"
            ]
            == "Uncertain"
        )

    # -------------------------------------------------------------------------
    # Save one CSV for this recording
    # -------------------------------------------------------------------------

    output_csv = (
        OUTPUT_DIR
        / f"{h5_path.stem}_predictions.csv"
    )

    results.to_csv(
        output_csv,
        index=False,
    )

    print(
        "Saved predictions:",
        output_csv.resolve(),
    )

    # -------------------------------------------------------------------------
    # Save skipped units, if any
    # -------------------------------------------------------------------------

    if skipped_units:

        skipped_csv = (
            OUTPUT_DIR
            / f"{h5_path.stem}_skipped_units.csv"
        )

        pd.DataFrame(
            skipped_units
        ).to_csv(
            skipped_csv,
            index=False,
        )

        print(
            "Saved skipped-unit log:",
            skipped_csv.resolve(),
        )

    # -------------------------------------------------------------------------
    # Print quick summary
    # -------------------------------------------------------------------------

    print("\nPrediction summary:")
    print(
        results[
            "forced_binary_prediction"
        ].value_counts()
    )

    if USE_UNCERTAINTY_CLASSIFICATION:

        print("\nHuman-in-the-loop summary:")
        print(
            results[
                "HITL_prediction"
            ].value_counts()
        )

    return results


# =============================================================================
# 10. CLASSIFY ALL NEW RECORDINGS
# =============================================================================

def main():
    """
    Load the trained model and classify all selected new recordings.
    """

    h5_files = get_h5_files()

    print(
        f"Found {len(h5_files)} new HDF5 file(s)."
    )

    model = load_trained_model()

    all_results = []

    for h5_path in h5_files:

        recording_results = classify_h5_file(
            model=model,
            h5_path=h5_path,
        )

        all_results.append(
            recording_results
        )

    # -------------------------------------------------------------------------
    # Save one combined CSV if multiple recordings were processed
    # -------------------------------------------------------------------------

    combined_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined_csv = (
        OUTPUT_DIR
        / "all_new_recording_predictions.csv"
    )

    combined_results.to_csv(
        combined_csv,
        index=False,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Total cells classified: "
        f"{len(combined_results)}"
    )

    print(
        "Combined predictions saved to:"
    )

    print(
        combined_csv.resolve()
    )

    if USE_UNCERTAINTY_CLASSIFICATION:

        review_count = int(
            combined_results[
                "needs_human_review"
            ].sum()
        )

        print(
            f"Cells sent for human review: "
            f"{review_count}"
        )


# =============================================================================
# 11. RUN
# =============================================================================

if __name__ == "__main__":
    main()
