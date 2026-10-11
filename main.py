#!/usr/bin/env python
"""Main entry point for Heart Disease pipeline.

Executes training using CLI arguments with at least 3 hyperparameters and a random seed,
falling back to default values if no arguments are provided.

Usage examples:
    # Run with all default parameters:
    python main.py

    # Run with custom hyperparameters and random seed:
    python main.py --n-estimators 200 --max-depth 10 --min-samples-split 4 --min-samples-leaf 2 --random-state 42

    # Register best model:
    python main.py --n-estimators 300 --max-depth 15 --register
"""

from heart_ccds.modeling.train import main

if __name__ == "__main__":
    main()
