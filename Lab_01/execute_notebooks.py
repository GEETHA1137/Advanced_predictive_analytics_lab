"""
Author : Geetha Priya S
Register Number : 23MID0021

This script executes all three project notebooks sequentially.
"""

import os
import subprocess
import sys

NOTEBOOKS = [
    "23MID0021_Lab01_Ames.ipynb",
    "23MID0021_Lab01_California.ipynb",
    "23MID0021_Lab01_UCI.ipynb"
]

def execute_notebook(notebook):
    print("=" * 60)
    print(f"Executing: {notebook}")
    print("=" * 60)

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        notebook,
        "--output",
        notebook
    ]

    subprocess.run(command, check=True)

    print(f"{notebook} completed successfully.\n")


def main():
    print("\nHouse Price Prediction Project")
    print("---------------------------------------")

    for notebook in NOTEBOOKS:
        if os.path.exists(notebook):
            execute_notebook(notebook)
        else:
            print(f"File not found: {notebook}")

    print("---------------------------------------")
    print("All notebooks executed successfully.")


if __name__ == "__main__":
    main()