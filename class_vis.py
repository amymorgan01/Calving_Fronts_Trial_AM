#!/usr/bin/env python3
"""
Class Distribution Analyzer

This script analyzes ground truth masks to visualize the distribution of classes.
It's designed to work with the dataset at: /gws/nopw/j04/iecdt/amorgan/data_copy/zones/train

Usage:
    python class_distribution_analyzer.py [--path /path/to/masks] [--samples N]
"""

import os
import sys
import argparse
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from torch.utils.data import Dataset
from PIL import Image
import tqdm

# Define class names and colors for visualization
CLASS_NAMES = ["Area", "Stone", "Glacier", "Ocean with Ice Melange"]
CLASS_COLORS = ["#8884d8", "#82ca9d", "#ffc658", "#ff8042"]

# Define mapping from grayscale values to class indices
# Modify these values based on your actual grayscale values
GRAYSCALE_MAPPING = {
    255: 0,  # White (Area)
    128: 1,  # Light gray (Stone)
    64: 2,   # Dark gray (Glacier)
    0: 3,    # Black (Ocean with Ice Melange)
}


class MaskAnalyzer:
    def __init__(self, mask_dir, max_samples=None):
        """
        Initialize the mask analyzer
        
        Args:
            mask_dir (str): Path to directory containing mask files
            max_samples (int, optional): Maximum number of samples to analyze
        """
        self.mask_dir = Path(mask_dir)
        self.max_samples = max_samples
        self.class_counts = np.zeros(4, dtype=np.int64)
        self.total_pixels = 0
        self.sample_count = 0
        
        # Find all mask files
        self.mask_files = self._find_mask_files()
        print(f"Found {len(self.mask_files)} mask files")
        
        # If max_samples is specified, randomly sample files
        if max_samples and max_samples < len(self.mask_files):
            self.mask_files = random.sample(self.mask_files, max_samples)
            print(f"Randomly selected {max_samples} mask files for analysis")
    
    def _find_mask_files(self):
        """Find all mask files in the specified directory"""
        # Check if directory exists
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.mask_dir}")
        
        # Look for common mask file extensions
        mask_files = []
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".pt"]:
            mask_files.extend(list(self.mask_dir.glob(f"**/*{ext}")))
        
        return mask_files
    
    def analyze_masks(self):
        """Analyze all mask files to compute class distribution"""
        print(f"Analyzing {len(self.mask_files)} mask files...")
        
        # First, let's analyze a few files to detect the unique grayscale values
        self._detect_grayscale_values()
        
        for mask_file in tqdm.tqdm(self.mask_files):
            try:
                self._analyze_single_mask(mask_file)
                self.sample_count += 1
            except Exception as e:
                print(f"Error analyzing {mask_file}: {e}")
        
        print(f"Successfully analyzed {self.sample_count} mask files")
        return self.get_results()
    
    def _detect_grayscale_values(self, sample_size=10):
        """Detect unique grayscale values in a sample of mask files"""
        unique_values = set()
        sample_files = self.mask_files[:min(sample_size, len(self.mask_files))]
        
        print("Detecting unique grayscale values in sample masks...")
        for mask_file in sample_files:
            try:
                # Load the mask image
                img = Image.open(mask_file)
                mask = np.array(img)
                
                # Get unique values
                file_unique = np.unique(mask)
                for val in file_unique:
                    unique_values.add(int(val))
            except Exception as e:
                print(f"Error detecting values in {mask_file}: {e}")
        
        print(f"Detected unique grayscale values: {sorted(unique_values)}")
        
        # Update the global mapping if needed
        if len(unique_values) <= 4:
            global GRAYSCALE_MAPPING
            sorted_values = sorted(unique_values)
            
            # Create a mapping from grayscale values to class indices
            for i, val in enumerate(sorted_values):
                if i < 4:  # Only map up to 4 classes
                    GRAYSCALE_MAPPING[val] = i
            
            print(f"Updated grayscale mapping: {GRAYSCALE_MAPPING}")
    
    def _analyze_single_mask(self, mask_file):
        """Analyze a single mask file"""
        # Load mask based on file extension
        ext = mask_file.suffix.lower()
        
        if ext in ['.pt', '.pth']:
            # PyTorch tensor
            mask = torch.load(mask_file)
            if torch.is_tensor(mask):
                # Handle different tensor formats
                if len(mask.shape) == 4:  # [1, C, H, W]
                    mask = mask.squeeze(0)
                if mask.shape[0] == 4:  # One-hot encoded [4, H, W]
                    mask = torch.argmax(mask, dim=0).numpy()
                else:
                    mask = mask.squeeze().numpy()
            else:
                raise ValueError(f"Unsupported tensor format: {type(mask)}")
                
        elif ext in ['.npy']:
            # NumPy array
            mask = np.load(mask_file)
            if mask.shape[0] == 4 and len(mask.shape) == 3:  # One-hot encoded [4, H, W]
                mask = np.argmax(mask, axis=0)
                
        elif ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            # Image file
            img = Image.open(mask_file)
            mask = np.array(img)
            
            # Map grayscale values to class indices
            mapped_mask = np.zeros_like(mask)
            
            # For each grayscale value in our mapping, set corresponding class index
            for gray_val, class_idx in GRAYSCALE_MAPPING.items():
                # Allow for some tolerance in grayscale values (±5)
                tolerance = 5
                lower_bound = max(0, gray_val - tolerance)
                upper_bound = min(255, gray_val + tolerance)
                
                if lower_bound == 0 and upper_bound == 255:
                    # Special case for full range - don't apply tolerance
                    mask_indices = (mask == gray_val)
                else:
                    # Apply tolerance range
                    mask_indices = ((mask >= lower_bound) & (mask <= upper_bound))
                
                mapped_mask[mask_indices] = class_idx
            
            mask = mapped_mask
        
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        # Count pixels for each class
        for cls in range(4):
            class_count = np.sum(mask == cls)
            self.class_counts[cls] += class_count
            
        self.total_pixels += mask.size
    
    def get_results(self):
        """Get analysis results"""
        if self.total_pixels == 0:
            raise ValueError("No pixels analyzed. Run analyze_masks() first.")
            
        class_freqs = self.class_counts / self.total_pixels
        
        results = {
            "sample_count": self.sample_count,
            "total_pixels": self.total_pixels,
            "class_counts": self.class_counts,
            "class_frequencies": class_freqs,
            "class_percentages": class_freqs * 100,
        }
        
        return results
    
    def calculate_class_weights(self, method='balanced'):
        """Calculate class weights based on distribution"""
        if self.total_pixels == 0:
            raise ValueError("No pixels analyzed. Run analyze_masks() first.")
            
        class_freqs = self.class_counts / self.total_pixels
        num_classes = len(class_freqs)
        
        if method == 'inverse':
            # Original method - inverse frequency
            class_weights = 1.0 / (class_freqs + 1e-8)
            
        elif method == 'balanced':
            # Balanced method - more moderate weights
            class_weights = 1.0 / (np.sqrt(class_freqs) + 1e-8)
            
        elif method == 'effective':
            # Effective number of samples
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, self.class_counts)
            class_weights = (1.0 - beta) / (effective_num + 1e-8)
            
        elif method == 'capped':
            # Capped inverse frequency with maximum ratio limit
            max_ratio = 5.0  # Maximum weight ratio between classes
            class_weights = 1.0 / (class_freqs + 1e-8)
            max_weight = np.max(class_weights)
            min_weight = np.min(class_weights)
            
            if max_weight / min_weight > max_ratio:
                # Scale down the highest weights
                class_weights = np.minimum(class_weights, max_ratio * min_weight)
        
        # Normalize weights so they sum to number of classes
        class_weights = class_weights / np.sum(class_weights) * num_classes
        
        return class_weights
    
    def plot_distribution(self, save_path=None):
        """Plot class distribution as bar chart and pie chart"""
        if self.total_pixels == 0:
            raise ValueError("No pixels analyzed. Run analyze_masks() first.")
        
        class_percentages = (self.class_counts / self.total_pixels * 100)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        ax1.bar(CLASS_NAMES, class_percentages, color=CLASS_COLORS)
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Class Distribution (Percentage)')
        for i, v in enumerate(class_percentages):
            ax1.text(i, v + 1, f"{v:.2f}%", ha='center')
        
        # Pie chart
        ax2.pie(class_percentages, labels=CLASS_NAMES, autopct='%1.1f%%', 
                shadow=True, startangle=90, colors=CLASS_COLORS)
        ax2.axis('equal')
        ax2.set_title('Class Distribution (Pie Chart)')
        
        plt.suptitle('Ground Truth Class Distribution', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_weights(self, save_path=None):
        """Plot different class weighting strategies"""
        if self.total_pixels == 0:
            raise ValueError("No pixels analyzed. Run analyze_masks() first.")
        
        # Calculate weights using different methods
        inverse_weights = self.calculate_class_weights('inverse')
        balanced_weights = self.calculate_class_weights('balanced')
        effective_weights = self.calculate_class_weights('effective')
        capped_weights = self.calculate_class_weights('capped')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart for weights
        x = np.arange(len(CLASS_NAMES))
        width = 0.2
        
        ax1.bar(x - 1.5*width, inverse_weights, width, label='Inverse')
        ax1.bar(x - 0.5*width, balanced_weights, width, label='Balanced')
        ax1.bar(x + 0.5*width, effective_weights, width, label='Effective')
        ax1.bar(x + 1.5*width, capped_weights, width, label='Capped')
        
        ax1.set_ylabel('Weight')
        ax1.set_title('Class Weights Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(CLASS_NAMES)
        ax1.legend()
        
        # Table for weight values
        cell_text = []
        for i in range(len(CLASS_NAMES)):
            cell_text.append([
                f"{inverse_weights[i]:.2f}",
                f"{balanced_weights[i]:.2f}", 
                f"{effective_weights[i]:.2f}",
                f"{capped_weights[i]:.2f}"
            ])
        
        ax2.axis('tight')
        ax2.axis('off')
        table = ax2.table(
            cellText=cell_text,
            rowLabels=CLASS_NAMES,
            colLabels=['Inverse', 'Balanced', 'Effective', 'Capped'],
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        ax2.set_title('Class Weight Values')
        
        plt.suptitle('Class Weighting Strategies', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved weights plot to {save_path}")
        
        plt.show()
    
    def visualize_mask_sample(self, num_samples=3, save_path=None):
        """Visualize a sample of masks with their mapped classes"""
        if len(self.mask_files) == 0:
            raise ValueError("No mask files found")
            
        # Select a few random samples
        sample_files = random.sample(self.mask_files, min(num_samples, len(self.mask_files)))
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = [axes]
            
        # Create a colormap for visualization
        cmap = ListedColormap(CLASS_COLORS)
        
        for i, mask_file in enumerate(sample_files):
            # Load original image
            img = Image.open(mask_file)
            orig_mask = np.array(img)
            
            # Map to class indices
            mapped_mask = np.zeros_like(orig_mask)
            for gray_val, class_idx in GRAYSCALE_MAPPING.items():
                tolerance = 5
                lower_bound = max(0, gray_val - tolerance)
                upper_bound = min(255, gray_val + tolerance)
                
                if lower_bound == 0 and upper_bound == 255:
                    # Special case for full range
                    mask_indices = (orig_mask == gray_val)
                else:
                    # Apply tolerance range
                    mask_indices = ((orig_mask >= lower_bound) & (orig_mask <= upper_bound))
                
                mapped_mask[mask_indices] = class_idx
            
            # Plot original mask
            axes[i][0].imshow(orig_mask, cmap='gray')
            axes[i][0].set_title(f"Original Mask: {mask_file.name}")
            axes[i][0].axis('off')
            
            # Plot mapped mask with class colors
            axes[i][1].imshow(mapped_mask, cmap=cmap, vmin=0, vmax=3)
            axes[i][1].set_title("Mapped Classes")
            axes[i][1].axis('off')
            
            # Add a small legend
            for j in range(4):
                axes[i][1].plot([], [], color=CLASS_COLORS[j], label=f"{CLASS_NAMES[j]} (Class {j})")
            axes[i][1].legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved mask visualization to {save_path}")
            
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze class distribution in mask files')
    parser.add_argument('--path', type=str, default='/gws/nopw/j04/iecdt/amorgan/data_copy/zones/train',
                        help='Path to the directory containing mask files')
    parser.add_argument('--samples', type=int, default=None, 
                        help='Maximum number of samples to analyze (default: all)')
    parser.add_argument('--output', type=str, default='class_distribution.png',
                        help='Output file path for saving the plot')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize sample masks with class mapping')
    
    args = parser.parse_args()
    
    try:
        analyzer = MaskAnalyzer(args.path, args.samples)
        results = analyzer.analyze_masks()
        
        # Print results
        print("\nClass Distribution Analysis Results:")
        print(f"Total samples analyzed: {results['sample_count']}")
        print(f"Total pixels: {results['total_pixels']}")
        print("\nClass counts:")
        for i, name in enumerate(CLASS_NAMES):
            count = results['class_counts'][i]
            percentage = results['class_percentages'][i]
            print(f"{name} (Class {i}): {count} pixels ({percentage:.2f}%)")
        
        print("\nGrayscale to class mapping used:")
        for gray_val, class_idx in GRAYSCALE_MAPPING.items():
            print(f"Grayscale value {gray_val} → Class {class_idx} ({CLASS_NAMES[class_idx]})")
        
        print("\nClass weights:")
        inverse_weights = analyzer.calculate_class_weights('inverse')
        balanced_weights = analyzer.calculate_class_weights('balanced')
        effective_weights = analyzer.calculate_class_weights('effective')
        capped_weights = analyzer.calculate_class_weights('capped')
        
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name} (Class {i}):")
            print(f"  Inverse: {inverse_weights[i]:.4f}")
            print(f"  Balanced: {balanced_weights[i]:.4f}")
            print(f"  Effective: {effective_weights[i]:.4f}")
            print(f"  Capped: {capped_weights[i]:.4f}")
        
        # Generate plots
        analyzer.plot_distribution(save_path=args.output)
        analyzer.plot_weights(save_path='class_weights.png')
        
        # Visualize sample masks if requested
        if args.visualize:
            analyzer.visualize_mask_sample(save_path='mask_visualization.png')
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())