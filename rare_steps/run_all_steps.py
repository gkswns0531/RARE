#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

def run_step(step_script, description):
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, step_script], 
                              capture_output=False, 
                              text=True, 
                              cwd=Path(__file__).parent)
        if result.returncode != 0:
            print(f"Error: {description} failed with return code {result.returncode}")
            return False
        print(f"Success: {description} completed")
        return True
    except Exception as e:
        print(f"Error running {description}: {str(e)}")
        return False

def main():
    steps = [
        ("run_step1_pdf_parsing.py", "Step 1: PDF Parsing"),
        ("run_step2_text_chunking.py", "Step 2: Text Chunking"), 
        ("run_step3_atomic_extraction.py", "Step 3: Atomic Information Extraction"),
        ("run_step4_best_selection.py", "Step 4: Best Information Selection"),
        ("run_step5_embedding.py", "Step 5: Embedding Generation"),
        ("run_step6_redundancy.py", "Step 6: Redundancy Detection"),
        ("run_step7_multihop_questions.py", "Step 7: Multi-hop Question Generation")
    ]
    
    print("RARE - Complete Pipeline Execution")
    print("All scripts use relative paths from rare_steps directory")
    print("Make sure your input PDFs are in examples/ directory")
    
    current_dir = Path(__file__).parent
    if not (current_dir / ".." / "examples").resolve().exists():
        print(f"\nWarning: examples directory not found!")
        print("Please create it and add your PDF files:")
        print("  mkdir -p examples")
        print("  cp your_documents.pdf examples/")
        return
    
    success_count = 0
    
    for step_script, description in steps:
        if run_step(step_script, description):
            success_count += 1
        else:
            print(f"\nPipeline stopped at {description}")
            break
    
    print(f"\n{'='*60}")
    print(f"Pipeline Summary: {success_count}/{len(steps)} steps completed")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
