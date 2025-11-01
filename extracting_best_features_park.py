#Stacking Ensemble → combines Random Forest, SVM, AdaBoost, XGBoost for final prediction.

import os
import pandas as pd
import parselmouth
from parselmouth.praat import call
from glob import glob
import pickle
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

# ---------------------- FEATURE EXTRACTION ----------------------
def extract_voice_features(file_path):
    try:
        snd = parselmouth.Sound(file_path)
        duration = snd.get_total_duration()
        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        intensity = call(snd, "To Intensity", 75, 0.0, True)

        meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")
        stdF0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        minF0 = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        maxF0 = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        pitch_range = maxF0 - minF0

        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        hnr = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        meanHNR = call(hnr, "Get mean", 0, 0)

        mean_intensity = call(intensity, "Get mean", 0, 0, "energy")
        std_intensity = call(intensity, "Get standard deviation", 0, 0)

        return {
            "meanF0": meanF0,
            "stdF0": stdF0,
            "pitch_range": pitch_range,
            "jitter_local": jitter_local,
            "shimmer_local": shimmer_local,
            "meanHNR": meanHNR,
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "duration": duration
        }
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None


# ---------------------- PROCESS DATA ----------------------
def process_audio_dataset(pd_folder, hc_folder, output_csv="optimized_features.csv"):
    all_data = []

    pd_folders = pd_folder.split(";")
    hc_folders = hc_folder.split(";")

    for label, folders in [(1, pd_folders), (0, hc_folders)]:
        label_name = "PD" if label == 1 else "HC"
        print(f"\nProcessing {label_name} files...")
        for folder in folders:
            files = glob(os.path.join(folder, "*.wav"))
            for i, f in enumerate(files, 1):
                print(f"Processing {label_name} file {i}/{len(files)}: {os.path.basename(f)}")
                features = extract_voice_features(f)
                if features:
                    features["filename"] = os.path.basename(f)
                    features["status"] = label
                    all_data.append(features)

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Feature extraction complete! Total usable samples: {len(df)}")
    print(f"📁 Saved to: {os.path.abspath(output_csv)}")
    return df


# ---------------------- MODEL TRAINING ----------------------
def train_optimized_model(csv_file="optimized_features.csv"):
    df = pd.read_csv(csv_file)
    if "filename" in df.columns:
        df = df.drop("filename", axis=1)

    print(f"\nDataset loaded: {len(df)} samples")
    print(f"Parkinson's: {sum(df['status'] == 1)}, Healthy: {sum(df['status'] == 0)}")

    X = df.drop("status", axis=1)
    y = df["status"]

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Remove low-variance features
    selector = VarianceThreshold(threshold=0.0001)
    X = selector.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Model tuning ----
    rf_params = {"n_estimators": [100, 200], "max_depth": [8, 10, 12]}
    rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"Best RandomForest Params: {rf.best_params_}")

    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    xgb = XGBClassifier(n_estimators=100, max_depth=5, eval_metric="logloss",
                        use_label_encoder=False, random_state=42)

    # ---- Stacking ensemble ----
    stack_model = StackingClassifier(
        estimators=[("rf", rf.best_estimator_), ("svm", svm), ("ada", ada)],
        final_estimator=xgb,
        cv=5,
        n_jobs=-1
    )

    stack_model.fit(X_train, y_train)

    # --- Training vs Validation accuracy ---
    train_acc = stack_model.score(X_train, y_train)
    y_pred = stack_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {test_acc:.4f}")

    if train_acc - test_acc > 0.1:
        print("⚠️ Possible overfitting detected — training accuracy much higher than validation accuracy.")
    else:
        print("✅ Model shows good generalization (no strong overfitting detected).")

    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"]))

    with open("optimized_best_model.pkl", "wb") as f:
        pickle.dump({
            "model": stack_model,
            "scaler": scaler,
            "imputer": imputer
        }, f)

    print("\n✓ Saved best model as 'optimized_best_model.pkl'")


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    df = process_audio_dataset(
        pd_folder="PD_AH;PD_AH_aug",
        hc_folder="HC_AH;HC_AH_aug"
    )
    train_optimized_model()

