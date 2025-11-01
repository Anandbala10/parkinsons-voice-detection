#Parkinson’s Voice Analysis → extracts acoustic biomarkers (pitch, jitter, shimmer, HNR)
#from speech and trains Random Forest, XGBoost, SVM, and AdaBoost models to detect Parkinson’s Disease.
import os
import pandas as pd
from glob import glob
import parselmouth
from parselmouth.praat import call
import pickle
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------------------
def extract_voice_features(file_path):
    try:
        snd = parselmouth.Sound(file_path)
        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)

        meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")
        stdF0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        meanHNR = call(hnr, "Get mean", 0, 0)

        return {
            "meanF0": meanF0,
            "stdF0": stdF0,
            "jitter_local": jitter_local,
            "shimmer_local": shimmer_local,
            "meanHNR": meanHNR,
        }
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None


# -------------------------------------------------------------
# MULTI-FOLDER DATASET HANDLER
# -------------------------------------------------------------
def process_audio_dataset(pd_folder, hc_folder, output_csv="parkinsons_voice_features.csv"):
    all_data = []

    pd_folders = pd_folder.split(";")
    hc_folders = hc_folder.split(";")

    print("Processing Parkinson's Disease files...")
    for folder in pd_folders:
        files = glob(os.path.join(folder, "*.wav"))
        for i, file_path in enumerate(files, 1):
            print(f"Processing PD file {i}/{len(files)}: {os.path.basename(file_path)}")
            features = extract_voice_features(file_path)
            if features:
                features["filename"] = os.path.basename(file_path)
                features["status"] = 1
                all_data.append(features)

    print("\nProcessing Healthy Control files...")
    for folder in hc_folders:
        files = glob(os.path.join(folder, "*.wav"))
        for i, file_path in enumerate(files, 1):
            print(f"Processing HC file {i}/{len(files)}: {os.path.basename(file_path)}")
            features = extract_voice_features(file_path)
            if features:
                features["filename"] = os.path.basename(file_path)
                features["status"] = 0
                all_data.append(features)

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Feature extraction complete!")
    print(f"📊 Total samples: {len(df)} | Saved to: {os.path.abspath(output_csv)}")
    return df


# -------------------------------------------------------------
# MODEL TRAINING & COMPARISON
# -------------------------------------------------------------
FEATURE_NAMES = ["meanF0", "stdF0", "jitter_local", "shimmer_local", "meanHNR"]

def train_and_compare_models(csv_file="parkinsons_voice_features.csv"):
    print("\n" + "=" * 70)
    print("TRAINING AND COMPARING ML MODELS ON (AUGMENTED) DATASET")
    print("=" * 70)

    df = pd.read_csv(csv_file)

    if "filename" in df.columns:
        df = df.drop("filename", axis=1)

    print(f"\nDataset loaded: {len(df)} samples")
    print(f"Parkinson's: {sum(df['status'] == 1)}, Healthy: {sum(df['status'] == 0)}")

    X = df[FEATURE_NAMES]
    y = df["status"]

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, max_depth=5,
                                 use_label_encoder=False, eval_metric="logloss"),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=100, learning_rate=1.0, random_state=42, algorithm='SAMME'
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'-' * 70}")
        print(f"Training {name}...")
        print(f"{'-' * 70}")

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"])

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring="accuracy")
        results[name] = {
            "accuracy": acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": cm,
            "report": report
        }

        print(f"\nAccuracy: {acc:.4f}")
        print(f"Cross-Validation (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Test Accuracy':<20} {'CV Accuracy':<20}")
    print("-" * 70)
    for name, res in results.items():
        print(f"{name:<20} {res['accuracy']:<20.4f} {res['cv_mean']:<20.4f}")

    best_name = max(results, key=lambda k: results[k]['accuracy'])
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_name} (Accuracy: {results[best_name]['accuracy']:.4f})")
    print("=" * 70)

    # Save model
    with open("best_augmented_model.pkl", "wb") as f:
        pickle.dump({
            "model": models[best_name],
            "scaler": scaler,
            "imputer": imputer,
            "feature_names": FEATURE_NAMES
        }, f)

    print("\n✓ Best model saved as 'best_augmented_model.pkl'")


# -------------------------------------------------------------
# MAIN EXECUTION (Full Pipeline)
# -------------------------------------------------------------
if __name__ == "__main__":
    df = process_audio_dataset(
        pd_folder="PD_AH;PD_AH_aug",
        hc_folder="HC_AH;HC_AH_aug",
        output_csv="parkinsons_voice_features.csv"
    )
    train_and_compare_models("parkinsons_voice_features.csv")
