import os
import pandas as pd
pd.set_option('display.max_columns', None)
import parselmouth
from parselmouth.praat import call
import numpy as np
from glob import glob
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# canonical feature order used everywhere
FEATURE_NAMES = [
    'jitter_local', 'jitter_abs', 'jitter_RAP', 'jitter_PPQ', 'jitter_DDP',
    'shimmer_local', 'shimmer_dB', 'shimmer_APQ3', 'shimmer_APQ5',
    'shimmer_APQ11', 'shimmer_DDA', 'HNR'
]


def extract_voice_features(sound_file):
    """
    Extracts voice features, handling potential errors gracefully.
    Returns dict with keys in FEATURE_NAMES (values may be None).
    """
    try:
        sound = parselmouth.Sound(sound_file)
        # initialize
        jitter_local = jitter_abs = jitter_rap = jitter_ppq = jitter_ddp = None
        shimmer_local = shimmer_db = shimmer_apq3 = shimmer_apq5 = shimmer_apq11 = shimmer_dda = None
        hnr = None

        # pitch-dependent features
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ddp = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_db = call([sound, point_process], "Get shimmer (local, dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq11 = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_dda = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except Exception:
            # keep None if pitch-dependent extraction fails
            pass

        # HNR
        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
        except Exception:
            pass

        features = {
            'jitter_local': jitter_local, 'jitter_abs': jitter_abs, 'jitter_RAP': jitter_rap,
            'jitter_PPQ': jitter_ppq, 'jitter_DDP': jitter_ddp, 'shimmer_local': shimmer_local,
            'shimmer_dB': shimmer_db, 'shimmer_APQ3': shimmer_apq3, 'shimmer_APQ5': shimmer_apq5,
            'shimmer_APQ11': shimmer_apq11, 'shimmer_DDA': shimmer_dda, 'HNR': hnr
        }

        # replace NaN/inf with None
        for k, v in list(features.items()):
            if v is not None and (np.isnan(v) or np.isinf(v)):
                features[k] = None

        # ensure keys present and in canonical order
        return {k: features.get(k, None) for k in FEATURE_NAMES}

    except Exception as e:
        print(f"A general error occurred processing {sound_file}: {e}")
        return {k: None for k in FEATURE_NAMES}


def process_audio_dataset(pd_folder='PD_AH', hc_folder='HC_AH', output_csv='parkinsons_voice_features.csv'):
    all_data = []

    print(f"Processing Parkinson's Disease files from {pd_folder}...")
    pd_files = glob(os.path.join(pd_folder, '*.wav'))
    for i, file_path in enumerate(pd_files, 1):
        print(f"Processing PD file {i}/{len(pd_files)}: {os.path.basename(file_path)}")
        features = extract_voice_features(file_path)
        features['filename'] = os.path.basename(file_path)
        features['status'] = 1
        all_data.append(features)

    print(f"\nProcessing Healthy Control files from {hc_folder}...")
    hc_files = glob(os.path.join(hc_folder, '*.wav'))
    for i, file_path in enumerate(hc_files, 1):
        print(f"Processing HC file {i}/{len(hc_files)}: {os.path.basename(file_path)}")
        features = extract_voice_features(file_path)
        features['filename'] = os.path.basename(file_path)
        features['status'] = 0
        all_data.append(features)

    column_order = ['filename'] + FEATURE_NAMES + ['status']
    df = pd.DataFrame(all_data, columns=column_order)
    df.to_csv(output_csv, index=False)

    print(f"\nFeatures saved to {output_csv}")
    print("\nDataset Summary:")
    print(f"Total files processed: {len(df)}")
    print(f"Parkinson's Disease files: {len(df[df['status'] == 1])}")
    print(f"Healthy Control files: {len(df[df['status'] == 0])}")
    print(f"Files with at least one missing feature: {df.isnull().any(axis=1).sum()}")

    return df


def train_and_compare_models(csv_file='parkinsons_voice_features.csv'):
    """
    Trains and compares Random Forest, XGBoost, and SVM models.
    Returns the best model, scaler, imputer, model_name, and results dict.
    """
    print("\n" + "=" * 70)
    print("TRAINING AND COMPARING ML MODELS")
    print("=" * 70)

    # Load dataset
    df = pd.read_csv(csv_file)

    # Remove filename column
    if 'filename' in df.columns:
        df = df.drop('filename', axis=1)

    print(f"\nDataset loaded: {len(df)} samples (before imputation)")
    print(f"Parkinson's: {sum(df['status'] == 1)}, Healthy: {sum(df['status'] == 0)}")

    # Separate features and target
    X = df[FEATURE_NAMES].copy()
    y = df['status'].copy()

    # Impute missing values (mean)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Ensure at least two classes exist
    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must contain at least two classes (0 and 1).")

    # Split data (try stratify; fallback to non-stratified if stratify fails)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42, stratify=None
        )

    # Scale features (fit on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models (SVM with probability=True to allow predict_proba)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, max_depth=5, use_label_encoder=False, eval_metric='logloss'),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    }

    results = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n{'-' * 70}")
        print(f"Training {model_name}...")
        print(f"{'-' * 70}")

        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Healthy', "Parkinson's"])

        # Cross-validation score (on scaled train data)
        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception:
            cv_mean = None
            cv_std = None
            cv_scores = []

        # Store results
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_accuracy': cv_mean,
            'cv_std': cv_std,
            'confusion_matrix': cm,
            'classification_report': report
        }

        # Print results
        print(f"\nAccuracy: {accuracy:.4f}")
        if cv_mean is not None:
            print(f"Cross-Validation Accuracy (5-fold): {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(report)

    # Comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Test Accuracy':<20} {'CV Accuracy':<20}")
    print("-" * 70)
    for model_name, result in results.items():
        cv_display = f"{result['cv_accuracy']:.4f}" if result['cv_accuracy'] is not None else "N/A"
        print(f"{model_name:<20} {result['accuracy']:<20.4f} {cv_display:<20}")

    # Select best model based on test accuracy (tie-breaker: CV if available)
    best_model_name = max(results.keys(), key=lambda k: (results[k]['accuracy'], results[k].get('cv_accuracy') or 0))
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    print("=" * 70)

    return best_model, scaler, imputer, best_model_name, results


def save_model(model, scaler, imputer, model_name, filename='best_parkinsons_model.pkl'):
    """
    Saves model, scaler, imputer, and metadata to disk.
    """
    model_data = {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'model_name': model_name,
        'feature_names': FEATURE_NAMES
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n✓ Best model and preprocessing saved to: {filename}")


def load_model(filename='best_parkinsons_model.pkl'):
    """
    Loads saved model + preprocessing from disk.
    """
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['scaler'], model_data['imputer'], model_data['model_name'], model_data['feature_names']


def predict_new_sample(audio_file, model_file='best_parkinsons_model.pkl'):
    """
    Predicts Parkinson's disease for a new audio sample.
    Returns prediction (0 or 1) or None on error.
    """
    print("\n" + "=" * 70)
    print("PREDICTING NEW AUDIO SAMPLE")
    print("=" * 70)
    print(f"Audio file: {audio_file}")

    if not os.path.exists(audio_file):
        print(f"ERROR: File '{audio_file}' not found!")
        return None

    # Load model + preprocessing
    print(f"\nLoading model from {model_file}...")
    model, scaler, imputer, model_name, feature_names = load_model(model_file)
    print(f"Model loaded: {model_name}")

    # Extract features
    print("\nExtracting voice features...")
    features = extract_voice_features(audio_file)

    # Build feature vector in proper order (fill missing with np.nan for imputer)
    fv = []
    missing = []
    for fn in feature_names:
        val = features.get(fn, None)
        if val is None:
            missing.append(fn)
            fv.append(np.nan)
        else:
            fv.append(val)

    if missing:
        print(f"\nNote: Could not extract {len(missing)} features: {missing}. They will be imputed.")

    feature_vector = np.array(fv).reshape(1, -1)

    # Impute missing values, then scale
    feature_vector_imputed = imputer.transform(feature_vector)
    feature_vector_scaled = scaler.transform(feature_vector_imputed)

    # Predict
    prediction = model.predict(feature_vector_scaled)[0]

    # Probability if available
    prob_healthy = prob_parkinsons = None
    try:
        probs = model.predict_proba(feature_vector_scaled)[0]
        prob_healthy, prob_parkinsons = probs[0], probs[1]
    except Exception:
        pass

    # Print results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"Prediction: {'PARKINSONS DISEASE' if prediction == 1 else 'HEALTHY'}")

    if prob_healthy is not None:
        print(f"\nConfidence:")
        print(f"  Healthy: {prob_healthy*100:.2f}%")
        print(f"  Parkinson's: {prob_parkinsons*100:.2f}%")

    print("\nExtracted / Imputed Features:")
    imputed_vector = feature_vector_imputed.flatten()
    for i, fn in enumerate(feature_names):
        print(f"  {fn}: {imputed_vector[i]:.6f} {'(imputed)' if np.isnan(fv[i]) else ''}")

    print("=" * 70)
    return prediction


if __name__ == "__main__":

    # 1) Extract features into CSV (skips if folders empty)
    print("STEP 1: FEATURE EXTRACTION")
    df = process_audio_dataset(
        pd_folder='PD_AH',
        hc_folder='HC_AH',
        output_csv='parkinsons_voice_features.csv'
    )

    print("\nFirst 5 rows of the extracted features:")
    print(df.head())

    print("\nBasic statistics of numerical features:")
    print(df.describe())

    # 2) Train & compare models
    print("\n\nSTEP 2: MODEL TRAINING AND COMPARISON")
    best_model, scaler, imputer, best_model_name, results = train_and_compare_models(
        csv_file='parkinsons_voice_features.csv'
    )

    # 3) Save the best model and preprocessors
    print("\n\nSTEP 3: SAVING BEST MODEL")
    save_model(best_model, scaler, imputer, best_model_name, filename='best_parkinsons_model.pkl')

    # 4) Usage: predict on a new sample
    print("\n\nSTEP 4: PREDICTION ON NEW SAMPLE")
    print("\nTo predict on a new audio file, call:")
    print("prediction = predict_new_sample('path/to/new_sample.wav')")
    # Example (uncomment if you have a 'new_sample.wav' in same folder)
    # prediction = predict_new_sample('new_sample.wav')
