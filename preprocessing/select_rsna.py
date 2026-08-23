from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_RSNA = PROJECT_ROOT / "data" / "raw" / "rsna"

LABEL_FILE = RAW_RSNA / "stage_2_train_labels.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rsna_selection"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "selected_rsna.csv"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

NORMAL_COUNT = 500
PNEUMONIA_COUNT = 500

RANDOM_SEED = 42


# --------------------------------------------------
# Load labels
# --------------------------------------------------

def load_labels() -> pd.DataFrame:

    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"RSNA label file not found:\n"
            f"{LABEL_FILE}"
        )

    df = pd.read_csv(LABEL_FILE)

    required_columns = {
        "patientId",
        "Target",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df


# --------------------------------------------------
# Convert RSNA labels to patient-level labels
# --------------------------------------------------

def create_patient_labels(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert image-level RSNA annotations
    into one label per patient.

    If any annotation for a patient has
    Target=1, the patient is classified
    as Pneumonia.

    Otherwise the patient is classified
    as Non-Pneumonia.
    """

    patient_labels = (
        df.groupby("patientId")["Target"]
        .max()
        .reset_index()
    )

    patient_labels["label"] = (
        patient_labels["Target"]
        .map(
            {
                0: "NON_PNEUMONIA",
                1: "PNEUMONIA",
            }
        )
    )

    return patient_labels


# --------------------------------------------------
# Select balanced subset
# --------------------------------------------------

def select_subset(
    patient_labels: pd.DataFrame,
) -> pd.DataFrame:

    pneumonia = patient_labels[
        patient_labels["label"]
        == "PNEUMONIA"
    ]

    non_pneumonia = patient_labels[
        patient_labels["label"]
        == "NON_PNEUMONIA"
    ]

    print("\nAvailable cases:")
    print(
        f"  Pneumonia:     {len(pneumonia)}"
    )
    print(
        f"  Non-Pneumonia: {len(non_pneumonia)}"
    )

    if len(pneumonia) < PNEUMONIA_COUNT:
        raise ValueError(
            "Not enough pneumonia cases."
        )

    if len(non_pneumonia) < NORMAL_COUNT:
        raise ValueError(
            "Not enough non-pneumonia cases."
        )

    selected_pneumonia = pneumonia.sample(
        n=PNEUMONIA_COUNT,
        random_state=RANDOM_SEED,
    )

    selected_non_pneumonia = (
        non_pneumonia.sample(
            n=NORMAL_COUNT,
            random_state=RANDOM_SEED,
        )
    )

    selected = pd.concat(
        [
            selected_pneumonia,
            selected_non_pneumonia,
        ],
        ignore_index=True,
    )

    # Shuffle the final selection.
    selected = selected.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    return selected


# --------------------------------------------------
# Save selection
# --------------------------------------------------

def save_selection(
    selected: pd.DataFrame,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSelection saved to:\n"
        f"{OUTPUT_FILE}"
    )


# --------------------------------------------------
# Main selection function
# --------------------------------------------------

def select_rsna_subset():

    print("=" * 60)
    print("RSNA SUBSET SELECTION")
    print("=" * 60)

    df = load_labels()

    print(
        f"\nTotal annotation rows: {len(df)}"
    )

    patient_labels = (
        create_patient_labels(df)
    )

    print(
        f"Unique patients: "
        f"{len(patient_labels)}"
    )

    selected = select_subset(
        patient_labels
    )

    save_selection(selected)

    print("\nSelected dataset:")
    print(
        selected["label"]
        .value_counts()
    )

    print(
        f"\nTotal selected: "
        f"{len(selected)}"
    )

    return selected


# --------------------------------------------------
# Command-line execution
# --------------------------------------------------

if __name__ == "__main__":

    select_rsna_subset()