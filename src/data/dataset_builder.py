# src/data/databuilder.py

from pathlib import Path
import json
import random
import pandas as pd
from collections import defaultdict
import os
import tempfile
from io import BytesIO

# -------------------------
# Utility
# -------------------------

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# -------------------------
# Kaggle API Functions
# -------------------------

def list_kaggle_dataset_files(api, dataset_name: str, recursive: bool = True):
    """List all files in a Kaggle dataset"""
    try:
        files = api.dataset_list_files(dataset_name).files
        file_list = []
        for file in files:
            file_list.append(file.name)
        return file_list
    except Exception as e:
        print(f"Error listing files from {dataset_name}: {e}")
        return []

def build_ffpp_index_from_kaggle(api, dataset_name: str) -> pd.DataFrame:
    """Build FFPP index directly from Kaggle API"""
    print(f"Reading FaceForensics++ metadata from Kaggle ({dataset_name})...")
    
    rows = []
    
    # List all files in the dataset
    files = list_kaggle_dataset_files(api, dataset_name)
    
    # Filter MP4 files
    real_videos = [f for f in files if "original" in f.lower() and f.endswith(".mp4")]
    
    # Real videos
    for vid_path in real_videos:
        vid_name = Path(vid_path).stem
        rows.append({
            "video_path": f"kaggle://{dataset_name}/{vid_path}",
            "label": "real",
            "manipulation": "original",
            "identity": vid_name
        })
    
    # Fake videos by manipulation type
    manipulations = [
        "DeepFakeDetection", "Deepfakes", "Face2Face",
        "FaceShifter", "FaceSwap", "NeuralTextures"
    ]
    
    for manip in manipulations:
        fake_videos = [f for f in files if manip.lower() in f.lower() and f.endswith(".mp4")]
        for vid_path in fake_videos:
            vid_name = Path(vid_path).stem
            identity = vid_name.split("_")[0]
            rows.append({
                "video_path": f"kaggle://{dataset_name}/{vid_path}",
                "label": "fake",
                "manipulation": manip,
                "identity": identity
            })
    
    df = pd.DataFrame(rows)
    print(f"  Found {len(df)} videos in FaceForensics++")
    return df

def build_celebdf_index_from_kaggle(api, dataset_name: str) -> pd.DataFrame:
    """Build CelebDF index directly from Kaggle API"""
    print(f"Reading CelebDF metadata from Kaggle ({dataset_name})...")
    
    rows = []
    
    # List all files in the dataset
    files = list_kaggle_dataset_files(api, dataset_name)
    
    # Real videos
    real_videos = [f for f in files if "celeb-real" in f.lower() and f.endswith(".mp4")]
    for vid_path in real_videos:
        vid_name = Path(vid_path).stem
        rows.append({
            "video_path": f"kaggle://{dataset_name}/{vid_path}",
            "label": "real",
            "manipulation": "CelebDF",
            "identity": vid_name
        })
    
    # Fake videos
    fake_videos = [f for f in files if "celeb-synthesis" in f.lower() and f.endswith(".mp4")]
    for vid_path in fake_videos:
        vid_name = Path(vid_path).stem
        identity = vid_name.split("_")[0]
        rows.append({
            "video_path": f"kaggle://{dataset_name}/{vid_path}",
            "label": "fake",
            "manipulation": "CelebDF",
            "identity": identity
        })
    
    df = pd.DataFrame(rows)
    print(f"  Found {len(df)} videos in CelebDF")
    return df

# -------------------------
# FaceForensics++ Builder (Local)
# -------------------------

def build_ffpp_index(root: Path) -> pd.DataFrame:
    rows = []

    # Real videos
    for vid in (root / "original").glob("*.mp4"):
        rows.append({
            "video_path": str(vid),
            "label": "real",
            "manipulation": "original",
            "identity": vid.stem
        })

    # Fake videos
    manipulations = [
        "DeepFakeDetection", "Deepfakes", "Face2Face",
        "FaceShifter", "FaceSwap", "NeuralTextures"
    ]

    for manip in manipulations:
        for vid in (root / manip).glob("*.mp4"):
            identity = vid.stem.split("_")[0]
            rows.append({
                "video_path": str(vid),
                "label": "fake",
                "manipulation": manip,
                "identity": identity
            })

    df = pd.DataFrame(rows)
    if rows:
        return df
    else:
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=["video_path", "label", "manipulation", "identity"])

# -------------------------
# Celeb-DF Builder (Local)
# -------------------------

def build_celebdf_index(root: Path) -> pd.DataFrame:
    rows = []

    for vid in (root / "Celeb-real").glob("*.mp4"):
        rows.append({
            "video_path": str(vid),
            "label": "real",
            "manipulation": "CelebDF",
            "identity": vid.stem
        })

    for vid in (root / "Celeb-synthesis").glob("*.mp4"):
        identity = vid.stem.split("_")[0]
        rows.append({
            "video_path": str(vid),
            "label": "fake",
            "manipulation": "CelebDF",
            "identity": identity
        })

    df = pd.DataFrame(rows)
    if rows:
        return df
    else:
        return pd.DataFrame(columns=["video_path", "label", "manipulation", "identity"])

# -------------------------
# Task Construction
# -------------------------

def split_identities(df, seed=42):
    random.seed(seed)
    ids = df["identity"].unique().tolist()
    random.shuffle(ids)

    n = len(ids)
    return (
        ids[:int(0.6 * n)],
        ids[int(0.6 * n):int(0.8 * n)],
        ids[int(0.8 * n):]
    )

def build_tasks(df, prefix):
    tasks = defaultdict(list)
    for _, row in df.iterrows():
        name = f"{prefix}_{row['manipulation']}"
        tasks[name].append({
            "video_path": row["video_path"],
            "label": row["label"],
            "identity": row["identity"]
        })
    return tasks

# -------------------------
# Main Phase-1 Pipeline (Kaggle)
# -------------------------

def run_phase1_kaggle(
    api,
    ffpp_dataset: str,
    celebdf_dataset: str,
    output_root: Path
):
    """Run Phase 1 using Kaggle API (no downloads)"""
    print("Building tasks from Kaggle datasets (no local download)...\n")
    
    # ---------- FF++ ----------
    print("Processing FaceForensics++...")
    ffpp_df = build_ffpp_index_from_kaggle(api, ffpp_dataset)
    
    if len(ffpp_df) == 0:
        print("Warning: No videos found in FaceForensics++")
    else:
        train_ids, val_ids, test_ids = split_identities(ffpp_df)
        
        train_df = ffpp_df[ffpp_df.identity.isin(train_ids)]
        val_df   = ffpp_df[ffpp_df.identity.isin(val_ids)]
        test_df  = ffpp_df[ffpp_df.identity.isin(test_ids)]
        
        save_json(build_tasks(train_df, "FFPP"),
                  output_root / "ffpp/meta_train_tasks.json")
        save_json(build_tasks(val_df, "FFPP"),
                  output_root / "ffpp/meta_val_tasks.json")
        save_json(build_tasks(test_df, "FFPP"),
                  output_root / "ffpp/meta_test_tasks.json")
        
        print(f"  Created {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    
    # ---------- Celeb-DF ----------
    print("\nProcessing CelebDF...")
    celeb_df = build_celebdf_index_from_kaggle(api, celebdf_dataset)
    
    if len(celeb_df) == 0:
        print("Warning: No videos found in CelebDF")
    else:
        save_json(build_tasks(celeb_df, "CELEBDF"),
                  output_root / "celebdf/meta_test_tasks.json")
        
        print(f"  Created {len(celeb_df)} test samples")
    
    print("\nPhase 1 completed successfully")

# -------------------------
# Main Phase-1 Pipeline (Local)
# -------------------------

def run_phase1(
    ffpp_root: Path,
    celebdf_root: Path,
    output_root: Path
):
    """Run Phase 1 using local files"""
    # ---------- FF++ ----------
    ffpp_df = build_ffpp_index(ffpp_root)
    train_ids, val_ids, test_ids = split_identities(ffpp_df)

    train_df = ffpp_df[ffpp_df.identity.isin(train_ids)]
    val_df   = ffpp_df[ffpp_df.identity.isin(val_ids)]
    test_df  = ffpp_df[ffpp_df.identity.isin(test_ids)]

    save_json(build_tasks(train_df, "FFPP"),
              output_root / "ffpp/meta_train_tasks.json")
    save_json(build_tasks(val_df, "FFPP"),
              output_root / "ffpp/meta_val_tasks.json")
    save_json(build_tasks(test_df, "FFPP"),
              output_root / "ffpp/meta_test_tasks.json")

    # ---------- Celeb-DF ----------
    celeb_df = build_celebdf_index(celebdf_root)
    save_json(build_tasks(celeb_df, "CELEBDF"),
              output_root / "celebdf/meta_test_tasks.json")

    print("Phase 1 completed successfully")
