#G:\Text_Classification\disaster_llm\scripts\build_dataset.py
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.dataset_builder import CrisisBenchBuilder


builder = CrisisBenchBuilder(
    data_dir=r"G:\Text_Classification\disaster_llm\data\raw\crisisbench\data\all_data_en",
    output_path="data/interim/cleaned.csv"
)

builder.build()
