"""
Utility script to inspect .npz data files
"""

import numpy as np
import matplotlib.pyplot as plt


def inspect_npz_file(filepath):
    """
    Detailed inspection of .npz file contents.
    
    Args:
        filepath: Path to .npz file
    """
    data = np.load(filepath)
    
    print("\n" + "=" * 80)
    print(f"Inspecting: {filepath}")
    print("=" * 80)
    
    print(f"\nAvailable keys: {list(data.keys())}")
    
    for key in data.keys():
        arr = data[key]
        print(f"\n{'-' * 80}")
        print(f"Key: '{key}'")
        print(f"{'-' * 80}")
        print(f"  Shape:    {arr.shape}")
        print(f"  Dtype:    {arr.dtype}")
        
        # Calculate stats ignoring NaN
        print(f"  Min:      {np.nanmin(arr):.6f}")
        print(f"  Max:      {np.nanmax(arr):.6f}")
        print(f"  Mean:     {np.nanmean(arr):.6f}")
        print(f"  Std:      {np.nanstd(arr):.6f}")
        print(f"  NaN count: {np.sum(np.isnan(arr))}")
        
        # Show sample data
        if arr.ndim == 1:
            print(f"  Sample (first 10): {arr[:10]}")
        elif arr.ndim == 2:
            print(f"  Sample [0, :10]: {arr[0, :10]}")
        elif arr.ndim == 3:
            print(f"  Sample [0, :10, 0]: {arr[0, :10, 0]}")
    
    data.close()


def visualize_sample(filepath, sample_idx=0):
    """
    Visualize a sample EEG signal.
    
    Args:
        filepath: Path to .npz file
        sample_idx: Index of sample to visualize
    """
    data = np.load(filepath)
    signal = data['signal']  # Shape: (samples, time_points, channels)
    label = data['label']
    
    if sample_idx >= len(signal):
        sample_idx = 0
    
    sample = signal[sample_idx]  # Shape: (time_points, channels)
    
    # Handle NaN values by replacing with mean
    sample_clean = sample.copy()
    for ch in range(sample.shape[1]):
        channel_data = sample_clean[:, ch]
        if np.any(np.isnan(channel_data)):
            # Replace NaN with column mean
            mean_val = np.nanmean(channel_data)
            channel_data[np.isnan(channel_data)] = mean_val
            sample_clean[:, ch] = channel_data
    
    num_channels = sample.shape[1]
    
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 4 * num_channels))
    
    # Handle single channel case
    if num_channels == 1:
        axes = [axes]
    
    for ch in range(num_channels):
        axes[ch].plot(sample_clean[:, ch], linewidth=0.8, color='blue')
        axes[ch].set_ylabel(f'Channel {ch}\nAmplitude (μV)', fontsize=10)
        axes[ch].grid(True, alpha=0.3)
        axes[ch].set_xlim(0, len(sample_clean))
        
        # Set title only on first subplot
        if ch == 0:
            axes[ch].set_title(
                f'Sample {sample_idx} (Label: {int(label[sample_idx])}) - File: {filepath}',
                fontsize=12,
                fontweight='bold'
            )
        
        # Set xlabel only on last subplot
        if ch == num_channels - 1:
            axes[ch].set_xlabel('Time Points', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_filename = f'sample_visualization_{sample_idx}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to {output_filename}")
    
    # Also show statistics
    print(f"\nSignal Statistics (Sample {sample_idx}):")
    print(f"  Original NaN count: {np.sum(np.isnan(sample))}")
    for ch in range(num_channels):
        ch_data = sample_clean[:, ch]
        print(f"  Channel {ch}:")
        print(f"    Min:  {np.min(ch_data):.4f}")
        print(f"    Max:  {np.max(ch_data):.4f}")
        print(f"    Mean: {np.mean(ch_data):.4f}")
        print(f"    Std:  {np.std(ch_data):.4f}")
    
    plt.show()
    data.close()


def visualize_multiple_samples(filepath, num_samples=3):
    """
    Visualize multiple samples from the dataset.
    
    Args:
        filepath: Path to .npz file
        num_samples: Number of samples to visualize
    """
    data = np.load(filepath)
    signal = data['signal']
    label = data['label']
    
    num_samples = min(num_samples, len(signal))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 3 * num_samples))
    
    for i in range(num_samples):
        sample = signal[i]  # Shape: (time_points, channels)
        
        # Clean NaN
        sample_clean = sample.copy()
        for ch in range(sample.shape[1]):
            channel_data = sample_clean[:, ch]
            if np.any(np.isnan(channel_data)):
                mean_val = np.nanmean(channel_data)
                channel_data[np.isnan(channel_data)] = mean_val
                sample_clean[:, ch] = channel_data
        
        # Plot both channels
        for ch in range(2):
            if num_samples == 1:
                ax = axes[ch]
            else:
                ax = axes[i, ch]
            
            ax.plot(sample_clean[:, ch], linewidth=0.6, color='blue')
            ax.set_ylabel('Amplitude (μV)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, len(sample_clean))
            
            # Title
            if i == 0:
                ax.set_title(f'Channel {ch}', fontsize=10, fontweight='bold')
            
            # Add label info on the left
            if ch == 0:
                ax.text(-0.15, 0.5, f'Sample {i}\n(Label: {int(label[i])})',
                       transform=ax.transAxes,
                       fontsize=9,
                       verticalalignment='center',
                       horizontalalignment='right')
            
            # X-label on bottom row
            if i == num_samples - 1:
                ax.set_xlabel('Time Points', fontsize=9)
    
    plt.suptitle(f'Multiple Samples from {filepath}', fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_filename = f'multiple_samples_visualization.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Multiple samples visualization saved to {output_filename}")
    plt.show()
    data.close()


if __name__ == "__main__":
    print("=" * 80)
    print("EEG Dataset Inspection Tool")
    print("=" * 80)
    
    # Inspect all files
    files = [
        './data/S4b_train.npz',
        './data/X11b_train.npz',
        './data/S4b_test.npz',
        './data/X11b_test.npz'
    ]
    
    for filepath in files:
        try:
            inspect_npz_file(filepath)
        except FileNotFoundError:
            print(f"\n[ERROR] File not found: {filepath}")
        except Exception as e:
            print(f"\n[ERROR] Error loading {filepath}: {e}")
    
    # Visualize single sample
    print("\n" + "=" * 80)
    print("Generating sample visualization...")
    print("=" * 80)
    try:
        visualize_sample('./data/S4b_train.npz', sample_idx=0)
    except Exception as e:
        print(f"[ERROR] Could not visualize: {e}")
    
    # Visualize multiple samples
    print("\n" + "=" * 80)
    print("Generating multiple samples visualization...")
    print("=" * 80)
    try:
        visualize_multiple_samples('./data/S4b_train.npz', num_samples=3)
    except Exception as e:
        print(f"[ERROR] Could not visualize multiple samples: {e}")
    