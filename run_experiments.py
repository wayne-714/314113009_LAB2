"""
Batch experiment runner for EEG Classification (Windows compatible)
"""

import os
import subprocess
import time

def run_experiment(name, model, activation, dropout, lr=0.001, epochs=300):
    """Run a single experiment."""
    print("\n" + "=" * 80)
    print(f"Running Experiment: {name}")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Activation: {activation}")
    print(f"Dropout: {dropout}")
    print(f"Learning Rate: {lr}")
    print(f"Epochs: {epochs}")
    print("=" * 80 + "\n")
    
    # Build command
    cmd = [
        "python", "main.py",
        "-model", model,
        "-activation", activation,
        "-dropout", str(dropout),
        "-lr", str(lr),
        "-num_epochs", str(epochs)
    ]
    
    # Run experiment
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ {name} completed successfully in {elapsed_time/60:.2f} minutes")
    else:
        print(f"\n❌ {name} failed!")
    
    return result.returncode == 0


def main():
    """Run all experiments."""
    print("=" * 80)
    print("Starting EEG Classification Batch Experiments")
    print("=" * 80)
    
    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    # Define experiments
    experiments = [
        {
            "name": "Exp1: Baseline EEGNet (ELU, dropout=0.25)",
            "model": "eegnet",
            "activation": "elu",
            "dropout": 0.25,
            "lr": 0.001,
            "epochs": 300
        },
        {
            "name": "Exp2: EEGNet with ReLU",
            "model": "eegnet",
            "activation": "relu",
            "dropout": 0.25,
            "lr": 0.001,
            "epochs": 300
        },
        {
            "name": "Exp3: EEGNet with LeakyReLU",
            "model": "eegnet",
            "activation": "leakyrelu",
            "dropout": 0.25,
            "lr": 0.001,
            "epochs": 300
        },
        {
            "name": "Exp4: EEGNet with dropout=0.1",
            "model": "eegnet",
            "activation": "elu",
            "dropout": 0.1,
            "lr": 0.001,
            "epochs": 300
        },
        {
            "name": "Exp5: EEGNet with dropout=0.5",
            "model": "eegnet",
            "activation": "elu",
            "dropout": 0.5,
            "lr": 0.001,
            "epochs": 300
        },
        {
            "name": "Exp6: DeepConvNet (baseline)",
            "model": "deepconvnet",
            "activation": "elu",
            "dropout": 0.5,
            "lr": 0.001,
            "epochs": 300
        },
        {
            "name": "Exp7: DeepConvNet with dropout=0.3",
            "model": "deepconvnet",
            "activation": "elu",
            "dropout": 0.3,
            "lr": 0.001,
            "epochs": 300
        },
        {
            "name": "Exp8: DeepConvNet with ReLU",
            "model": "deepconvnet",
            "activation": "relu",
            "dropout": 0.5,
            "lr": 0.001,
            "epochs": 300
        }
    ]
    
    # Run experiments
    results = []
    total_start = time.time()
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'#' * 80}")
        print(f"# Experiment {i}/{len(experiments)}")
        print(f"{'#' * 80}\n")
        
        success = run_experiment(
            exp["name"],
            exp["model"],
            exp["activation"],
            exp["dropout"],
            exp["lr"],
            exp["epochs"]
        )
        
        results.append({
            "name": exp["name"],
            "success": success
        })
    
    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print(f"Total time: {total_time/3600:.2f} hours")
    print("\nResults Summary:")
    for i, res in enumerate(results, 1):
        status = "✅ Success" if res["success"] else "❌ Failed"
        print(f"  {i}. {res['name']}: {status}")
    print("=" * 80)
    print("Check './results/' for plots and './weights/' for models")
    print("=" * 80)


if __name__ == "__main__":
    main()