import os
import sys
import shutil
import random

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from augraphy import *

from core.utils.utils import project_path

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

paper_phase = [
    OneOf(
        [
            DelaunayTessellation(
                n_points_range=(500, 800),
                n_horizontal_points_range=(500, 800),
                n_vertical_points_range=(500, 800),
                noise_type="random",
                color_list="default",
                color_list_alternate="default",
            ),
            PatternGenerator(
                imgx=random.randint(256, 512),
                imgy=random.randint(256, 512),
                n_rotation_range=(10, 15),
                color="random",
                alpha_range=(0.35, 0.7),  # 더 강한 질감
            ),
            VoronoiTessellation(
                mult_range=(80, 120),               # 더 강한 분할
                num_cells_range=(800, 1500),        # 더 조밀하게
                noise_type="random",
                background_value=(180, 230),        # 더 어두운 톤
            ),
        ],
        p=1.0,
    ),
    AugmentationSequence(
        [
            NoiseTexturize(
                sigma_range=(20, 30),              # 더 뭉친 노이즈
                turbulence_range=(8, 15),          # 왜곡 세게
            ),
            BrightnessTexturize(
                texturize_range=(0.75, 0.9),       # 어두운 텍스처
                deviation=0.08,                    # 밝기 흔들림 증가
            ),
        ],
    ),
]

if __name__ == "__main__":
    seed = 2025

    src_data_path = os.path.join(project_path(), "data/train")
    dst_data_path = os.path.join(project_path(), f"data/train_augraphy_{seed}")
    src_csv_path = os.path.join(project_path(), "data/train.csv")
    dst_csv_path = os.path.join(project_path(), f"data/train_augraphy_{seed}.csv")
    os.makedirs(dst_data_path, exist_ok=True) 

    pipeline = AugraphyPipeline(paper_phase=paper_phase)

    df = pd.read_csv(src_csv_path)
    
    rows = []
    for fname, target in tqdm(df.values, "Augraphy"):
        src_path = os.path.join(src_data_path, fname)
        dst_path = os.path.join(dst_data_path, "aug_" + fname)

        img_np = np.array(Image.open(src_path))

        img = Image.open(src_path).convert("RGB")
        img_np = np.array(img)
        
        set_seed(seed)
        aug_np = pipeline(img_np)
        Image.fromarray(aug_np).save(dst_path)
        shutil.copy2(src_path, os.path.join(dst_data_path, fname))

        rows.append((fname, target))
        rows.append(("aug_" + fname, target))

    new_df = pd.DataFrame(rows, columns=["ID", "target"])
    new_df.to_csv(dst_csv_path, index=False)
    print(f"Done! Images ⇒ {dst_csv_path}\n   CSV    ⇒ {dst_csv_path}")
