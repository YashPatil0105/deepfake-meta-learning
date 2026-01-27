# scripts/run_phase1.py
"""
Generate Phase 1 task files from local datasets.
Expected structure:
  data/raw/faceforensics++/videos/original/*.mp4
  data/raw/faceforensics++/videos/DeepFakeDetection/*.mp4
  ...
  data/raw/celebdf/videos/Celeb-real/*.mp4
  data/raw/celebdf/videos/Celeb-synthesis/*.mp4
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_builder import run_phase1

# Define Local Paths
# Note: Based on standard project structure, videos are usually in 'raw/<dataset>/videos'
FFPP_ROOT = project_root / "data" / "raw" / "faceforensics++" / "videos"
CELEBDF_ROOT = project_root / "data" / "raw" / "celebdf" / "videos"

# Output paths
OUTPUT_ROOT = project_root / "data" / "splits"

print(f"Project Root: {project_root}")
print(f"FFPP Root: {FFPP_ROOT}")
print(f"CelebDF Root: {CELEBDF_ROOT}")
print(f"Output Root: {OUTPUT_ROOT}")

# Run Phase 1
print("\n" + "="*50)
print("Building meta-learning tasks from Local Data...")
print("="*50 + "\n")

try:
    # Basic validation
    if not FFPP_ROOT.exists():
        print(f"Warning: FFPP path does not exist: {FFPP_ROOT}")
        print("Please ensure you have unpacked the dataset into data/raw/faceforensics++/videos")
        
    if not CELEBDF_ROOT.exists():
        print(f"Warning: CelebDF path does not exist: {CELEBDF_ROOT}")
        print("Please ensure you have unpacked the dataset into data/raw/celebdf/videos")

    run_phase1(
        ffpp_root=FFPP_ROOT,
        celebdf_root=CELEBDF_ROOT,
        output_root=OUTPUT_ROOT
    )
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
