#G:\Text_Classification\disaster_llm\scripts\build_instructions.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.instruction_builder import InstructionBuilder

builder = InstructionBuilder(
    input_csv="data/interim/cleaned.csv",
    output_jsonl="data/processed/instructions.jsonl"
)

builder.build()
