import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# 1. Paths
audio_dir = "data/the-circor-digiscope-phonocardiogram-dataset-1.0.0/training/"
labels_path = "data/the-circor-digiscope-phonocardiogram-dataset-1.0.0/training_data.csv"

# 2. Load labels
df_labels = pd.read_csv(labels_path)
print("Labels file loaded. Columns:", df_labels.columns.tolist())
print("Murmurs distribution:")
print(df_labels['Murmurs'].value_counts())

# 3. Get all .wav files
audio_files = []
for root, dirs, filenames in os.walk(audio_dir):
    for f in filenames:
        if f.endswith('.wav'):
            audio_files.append(os.path.join(root, f))

print(f"Found {len(audio_files)} audio files")

# 4. Feature extraction function
def extract_advanced_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=5)
        
        # MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        rms = librosa.feature.rms(y=audio)[0]
        
        features = []
        features.extend(mfccs_mean)      # 13
        features.extend(mfccs_std)       # 13
        features.append(np.mean(spec_cent))
        features.append(np.std(spec_cent))
        features.append(np.mean(spec_bw))
        features.append(np.std(spec_bw))
        features.append(np.mean(spec_rolloff))
        features.append(np.std(spec_rolloff))
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        features.append(np.mean(rms))
        features.append(np.std(rms))    # 10
        
        return np.array(features)
    except Exception as e:
        return None

# 5. Extract features for all files
features_list = []
filenames_list = []

print("Extracting features... This will take 10-20 minutes.")

for file_path in tqdm(audio_files):
    fname = os.path.basename(file_path)
    features = extract_advanced_features(file_path)
    if features is not None:
        features_list.append(features)
        filenames_list.append(fname)

print(f"Extracted features for {len(features_list)} files")

# 6. Create DataFrame
df_features = pd.DataFrame(features_list)
df_features.columns = [f'MFCC_{i+1}' for i in range(13)] + [f'MFCC_std_{i+1}' for i in range(13)] + ['spec_cent_mean', 'spec_cent_std', 'spec_bw_mean', 'spec_bw_std', 'spec_rolloff_mean', 'spec_rolloff_std', 'zcr_mean', 'zcr_std', 'rms_mean', 'rms_std']

# 7. Extract Patient_ID from filename
def extract_patient_id(filename):
    parts = filename.split('_')
    if parts:
        return parts[0]  # Just return the first part (e.g., "14241")
    return None

# Extract Patient_ID and convert to integer
df_features['Patient_ID'] = [int(extract_patient_id(f)) for f in filenames_list]
df_features['Filename'] = filenames_list

# 8. Merge with labels
# Map Murmurs value to binary label
def map_murmur_to_label(value):
    # Murmurs: 0 = No murmur, 1 = Innocent, 2 = Pathological, 3 = Normal (some datasets)
    if value in [0, 3]:  # No murmur or Normal
        return 0
    else:  # 1 or 2 = murmur present
        return 1

label_map = df_labels.set_index('Patient_ID')['Murmurs'].to_dict()
df_features['Murmurs'] = df_features['Patient_ID'].map(label_map)

# Drop rows with missing labels
df_features = df_features.dropna(subset=['Murmurs'])
print(f"Matched {len(df_features)} files with labels")

# 9. Create binary label
df_features['Binary_Label'] = df_features['Murmurs'].apply(map_murmur_to_label)

print("\n📊 Murmurs distribution:")
print(df_features['Murmurs'].value_counts())
print("\n📊 Binary distribution (0 = Normal, 1 = Abnormal):")
print(df_features['Binary_Label'].value_counts())

# 10. Save to CSV
df_features.to_csv("heart_sound_features.csv", index=False)
print(f"\n✅ DONE! Saved {len(df_features)} samples to 'heart_sound_features.csv'")