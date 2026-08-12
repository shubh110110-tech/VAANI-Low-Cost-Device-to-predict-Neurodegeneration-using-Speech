import os
import zipfile
import shutil
import random
import json
import numpy as np
import librosa
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(r"C:\Users\HP\Desktop\VAANI")

PARKINSON_ZIP = BASE_DIR / "parkinsonsdataset.zip"
ALS_ZIP = BASE_DIR / "ALSdatasets.zip"

WORK_DIR = BASE_DIR / "vaani_training"
MODEL_DIR = BASE_DIR / "vaani_model"

SAMPLE_RATE = 16000
DURATION = 3.0
MAX_SAMPLES = int(SAMPLE_RATE * DURATION)

N_MFCC = 13
N_MELS = 26

ALS_LIMIT = 40

CLASS_NAMES = [
    "Healthy",
    "Parkinsons",
    "ALS"
]

LABELS = {
    "Healthy": 0,
    "Parkinsons": 1,
    "ALS": 2
}


def reset_directory(path):
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def extract_zip(zip_path, destination):
    print(f"\nExtracting:")
    print(zip_path)

    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(destination)

    print("Extraction complete.")


def find_all_wavs(directory):
    return [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() == ".wav"
    ]


def find_zip_files(directory):
    return [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() == ".zip"
    ]


def prepare_parkinsons_data():

    print("\n" + "=" * 70)
    print("PREPARING PARKINSON'S / HEALTHY DATA")
    print("=" * 70)

    if not PARKINSON_ZIP.exists():
        raise FileNotFoundError(
            f"Missing:\n{PARKINSON_ZIP}"
        )

    root = WORK_DIR / "parkinsons_root"
    reset_directory(root)

    extract_zip(PARKINSON_ZIP, root)

    nested_zips = find_zip_files(root)

    print("\nNested ZIP files found:")

    for z in nested_zips:
        print(" ", z)

    pd_zip = None
    hc_zip = None

    for z in nested_zips:

        name = z.name.lower()

        if "pd_ah" in name:
            pd_zip = z

        elif "hc_ah" in name:
            hc_zip = z

    if pd_zip is None:
        raise FileNotFoundError(
            "PD_AH.zip was not found inside parkinsonsdataset.zip"
        )

    if hc_zip is None:
        raise FileNotFoundError(
            "HC_AH.zip was not found inside parkinsonsdataset.zip"
        )

    pd_dir = WORK_DIR / "PD_AH"
    hc_dir = WORK_DIR / "HC_AH"

    reset_directory(pd_dir)
    reset_directory(hc_dir)

    extract_zip(pd_zip, pd_dir)
    extract_zip(hc_zip, hc_dir)

    pd_wavs = find_all_wavs(pd_dir)
    hc_wavs = find_all_wavs(hc_dir)

    print("\nParkinson WAV files:", len(pd_wavs))
    print("Healthy WAV files:", len(hc_wavs))

    if len(pd_wavs) == 0:
        raise RuntimeError(
            "No Parkinson WAV files found inside PD_AH.zip"
        )

    if len(hc_wavs) == 0:
        raise RuntimeError(
            "No Healthy WAV files found inside HC_AH.zip"
        )

    return hc_wavs, pd_wavs


def prepare_als_data():

    print("\n" + "=" * 70)
    print("PREPARING ALS DATA")
    print("=" * 70)

    if not ALS_ZIP.exists():
        raise FileNotFoundError(
            f"Missing:\n{ALS_ZIP}"
        )

    root = WORK_DIR / "als_root"
    reset_directory(root)

    extract_zip(ALS_ZIP, root)

    als_wavs = find_all_wavs(root)

    print("\nALS WAV files found:", len(als_wavs))

    if len(als_wavs) == 0:
        raise RuntimeError(
            "No WAV recordings were found inside ALSdatasets.zip"
        )

    random.seed(SEED)

    random.shuffle(als_wavs)

    als_wavs = als_wavs[:ALS_LIMIT]

    print("ALS recordings selected:", len(als_wavs))

    return als_wavs


def load_audio(path):

    try:

        audio, sr = librosa.load(
            str(path),
            sr=SAMPLE_RATE,
            mono=True
        )

        if audio is None or len(audio) == 0:
            return None

        audio = audio.astype(np.float32)

        max_value = np.max(np.abs(audio))

        if max_value > 0:
            audio = audio / max_value

        target_length = MAX_SAMPLES

        if len(audio) < target_length:

            audio = np.pad(
                audio,
                (0, target_length - len(audio))
            )

        else:

            audio = audio[:target_length]

        return audio

    except Exception as e:

        print(
            f"Could not load {path.name}: {e}"
        )

        return None


def extract_features(path):

    audio = load_audio(path)

    if audio is None:
        return None

    try:

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=512,
            hop_length=256
        )

        delta = librosa.feature.delta(mfcc)

        delta2 = librosa.feature.delta(
            mfcc,
            order=2
        )

        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=512,
            hop_length=256
        )

        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=512,
            hop_length=256
        )

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=512,
            hop_length=256
        )

        zero_crossing = librosa.feature.zero_crossing_rate(
            audio,
            hop_length=256
        )

        rms = librosa.feature.rms(
            y=audio,
            frame_length=512,
            hop_length=256
        )

        features = []

        for matrix in [
            mfcc,
            delta,
            delta2,
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            zero_crossing,
            rms
        ]:

            features.extend(
                np.mean(matrix, axis=1)
            )

            features.extend(
                np.std(matrix, axis=1)
            )

        features = np.asarray(
            features,
            dtype=np.float32
        )

        features = np.nan_to_num(
            features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return features

    except Exception as e:

        print(
            f"Feature extraction failed for {path.name}: {e}"
        )

        return None


def build_dataset():

    print("\n" + "=" * 70)
    print("BUILDING VAANI DATASET")
    print("=" * 70)

    healthy, parkinsons = prepare_parkinsons_data()

    als = prepare_als_data()

    all_paths = []
    all_labels = []

    for p in healthy:

        all_paths.append(p)
        all_labels.append(LABELS["Healthy"])

    for p in parkinsons:

        all_paths.append(p)
        all_labels.append(LABELS["Parkinsons"])

    for p in als:

        all_paths.append(p)
        all_labels.append(LABELS["ALS"])

    print("\nTotal recordings:")
    print("Healthy:", len(healthy))
    print("Parkinsons:", len(parkinsons))
    print("ALS:", len(als))
    print("Total:", len(all_paths))

    X = []
    y = []

    print("\nExtracting audio features...")

    for i, (path, label) in enumerate(
        zip(all_paths, all_labels)
    ):

        if i % 25 == 0:

            print(
                f"Processed {i}/{len(all_paths)}"
            )

        features = extract_features(path)

        if features is not None:

            X.append(features)
            y.append(label)

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    print("\nFeature matrix:", X.shape)
    print("Labels:", y.shape)

    for index, name in enumerate(CLASS_NAMES):

        print(
            f"{name}: {np.sum(y == index)}"
        )

    return X, y


def split_dataset(X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def scale_features(
    X_train,
    X_val,
    X_test
):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )

    return (
        X_train.astype(np.float32),
        X_val.astype(np.float32),
        X_test.astype(np.float32),
        scaler
    )


def save_scaler(scaler):

    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }

    with open(
        MODEL_DIR / "scaler.json",
        "w"
    ) as f:

        json.dump(
            scaler_data,
            f
        )


def build_model(input_size):

    inputs = tf.keras.Input(
        shape=(input_size,),
        name="audio_features"
    )

    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(
        0.25
    )(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.20
    )(x)

    outputs = tf.keras.layers.Dense(
        3,
        activation="softmax",
        name="prediction"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(
    X_train,
    y_train,
    X_val,
    y_val
):

    print("\n" + "=" * 70)
    print("TRAINING VAANI")
    print("=" * 70)

    model = build_model(
        X_train.shape[1]
    )

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(c): float(w)
        for c, w in zip(
            classes,
            weights
        )
    }

    print("\nClass weights:")
    print(class_weights)

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=100,
        batch_size=16,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print(
        f"\nAccuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )

    print("Confusion matrix:")
    print(matrix)

    np.savetxt(
        MODEL_DIR / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d"
    )

    return predictions


def save_keras_model(model):

    model.save(
        MODEL_DIR / "vaani_model.keras"
    )


def convert_to_tflite(model):

    print("\n" + "=" * 70)
    print("CONVERTING TO TFLITE")
    print("=" * 70)

    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    tflite_model = converter.convert()

    float_path = MODEL_DIR / "vaani_model.tflite"

    with open(
        float_path,
        "wb"
    ) as f:

        f.write(tflite_model)

    print(
        "TFLite model size:",
        len(tflite_model) / 1024,
        "KB"
    )

    return float_path


def representative_dataset_generator(X_train):

    sample_count = min(
        100,
        len(X_train)
    )

    for i in range(sample_count):

        sample = X_train[i:i + 1]

        yield [
            sample.astype(
                np.float32
            )
        ]


def convert_to_int8(
    model,
    X_train
):

    print("\n" + "=" * 70)
    print("INT8 QUANTIZATION")
    print("=" * 70)

    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = (
        lambda: representative_dataset_generator(
            X_train
        )
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    int8_model = converter.convert()

    path = MODEL_DIR / "vaani_model_int8.tflite"

    with open(
        path,
        "wb"
    ) as f:

        f.write(int8_model)

    print(
        "INT8 model size:",
        len(int8_model) / 1024,
        "KB"
    )

    return path


def convert_to_c_array(tflite_path):

    print("\n" + "=" * 70)
    print("CREATING ESP32 MODEL ARRAY")
    print("=" * 70)

    data = tflite_path.read_bytes()

    output = MODEL_DIR / "vaani_model_data.h"

    with open(
        output,
        "w"
    ) as f:

        f.write(
            "#ifndef VAANI_MODEL_DATA_H\n"
        )

        f.write(
            "#define VAANI_MODEL_DATA_H\n\n"
        )

        f.write(
            "const unsigned char vaani_model[] = {\n"
        )

        for i in range(
            0,
            len(data),
            12
        ):

            chunk = data[
                i:i + 12
            ]

            line = ", ".join(
                f"0x{b:02x}"
                for b in chunk
            )

            f.write(
                "    " + line + ",\n"
            )

        f.write(
            "};\n\n"
        )

        f.write(
            f"const unsigned int vaani_model_len = {len(data)};\n\n"
        )

        f.write(
            "#endif\n"
        )

    print(
        "Created:",
        output
    )


def save_metadata(
    X,
    scaler
):

    metadata = {

        "project": "VAANI",

        "task": "3-class voice screening",

        "classes": CLASS_NAMES,

        "sample_rate": SAMPLE_RATE,

        "duration_seconds": DURATION,

        "mfcc": N_MFCC,

        "mel_bands": N_MELS,

        "feature_count": int(
            X.shape[1]
        ),

        "als_limit": ALS_LIMIT,

        "seed": SEED,

        "note":
            "Research screening/classification model. "
            "Not a medical diagnosis or future-disease prediction."
    }

    with open(
        MODEL_DIR / "metadata.json",
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    save_scaler(
        scaler
    )


def main():

    print("\n")
    print("=" * 70)
    print("VAANI v2")
    print("3-CLASS EDGE AI VOICE SCREENING")
    print("=" * 70)

    reset_directory(
        WORK_DIR
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X, y = build_dataset()

    if len(X) == 0:
        raise RuntimeError(
            "No usable audio samples were extracted."
        )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(
        X,
        y
    )

    print("\nDataset split:")

    print(
        "Training:",
        len(X_train)
    )

    print(
        "Validation:",
        len(X_val)
    )

    print(
        "Testing:",
        len(X_test)
    )

    (
        X_train,
        X_val,
        X_test,
        scaler
    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    save_metadata(
        X,
        scaler
    )

    model, history = train_model(
        X_train,
        y_train,
        X_val,
        y_val
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    save_keras_model(
        model
    )

    tflite_path = convert_to_tflite(
        model
    )

    int8_path = convert_to_int8(
        model,
        X_train
    )

    convert_to_c_array(
        int8_path
    )

    print("\n" + "=" * 70)
    print("VAANI COMPLETE")
    print("=" * 70)

    print("\nHealthy      = HC_AH")
    print("Parkinsons   = PD_AH")
    print("ALS          = 40 ALS recordings")

    print(
        "\nModels saved in:"
    )

    print(
        MODEL_DIR
    )

    print("\nImportant files:")

    print(
        "vaani_model.keras"
    )

    print(
        "vaani_model.tflite"
    )

    print(
        "vaani_model_int8.tflite"
    )

    print(
        "vaani_model_data.h"
    )

    print(
        "scaler.json"
    )

    print(
        "metadata.json"
    )

    print("\nNext stage: ESP32 deployment.")


if __name__ == "__main__":
    main()
