"""
rm_background.py

A standalone script to analyze glacier segmentation dataset and identify images with
excessive background area. This tool helps identify problematic images before
modifying the main dataset class.

Usage:
    python analyze_glacier_dataset.py --data_dir /path/to/data --mode train --threshold 0.8
"""

import os
import cv2
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm


def analyze_dataset(data_dir, mode="train", threshold=0.8):
    """
    Analyzes a glacier dataset to identify images with excessive background.
    
    Args:
        data_dir (str): Root directory containing 'sar_images', 'zones', and 'fronts' folders
        mode (str): Dataset split to analyze ('train', 'val', or 'test')
        threshold (float): Maximum allowed background percentage (0-1.0)
        output_dir (str): Directory to save analysis results and visualizations
        move_files (bool): If True, will move background-heavy files to a separate directory
        visualize (bool): If True, will create visualizations of sample images
    
    Returns:
        pd.DataFrame: DataFrame with analysis results
    """
    print(f"Analyzing {mode} dataset at {data_dir}...")
    
    # Setup directories
    image_dir = os.path.join(data_dir, "sar_images", mode)
    mask_dir = os.path.join(data_dir, "zones", mode)
    front_dir = os.path.join(data_dir, "fronts", mode)
    
    # Get all files
    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))
    fronts = sorted(os.listdir(front_dir))
    
    assert len(images) == len(masks) == len(fronts), "Mismatch in dataset size!"
    
    print(f"Found {len(images)} images to analyze.")
    
    # Define the thresholds for class conversion
    thresholds = [0.0, 0.1, 0.3, 0.7, 1.1]
    
    # Create DataFrame to store results
    results = []
    
    # Analysis counters
    total_images = len(images)
    heavy_bg_count = 0
    
    # Process all images
    for i, (img_name, mask_name, front_name) in enumerate(tqdm(zip(images, masks, fronts), total=len(images))):
        # Load mask
        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0  # Normalize to 0-1
        
        # Calculate class distribution
        class_counts = np.zeros(len(thresholds) - 1)
        
        for k, thresh in enumerate(thresholds[:-1]):
            mask_k = np.logical_and(mask >= thresh, mask < thresholds[k+1])
            class_counts[k] = np.sum(mask_k)
        
        # Calculate percentages
        total_pixels = mask.size
        class_percentages = class_counts / total_pixels
        
        # Check if background (class 0) exceeds threshold
        is_bg_heavy = class_percentages[0] >= threshold
        
        if is_bg_heavy:
            heavy_bg_count += 1
        
        # Store results
        results.append({
            'image_name': img_name,
            'mask_name': mask_name,
            'front_name': front_name,
            'background_pct': class_percentages[0],
            'class1_pct': class_percentages[1],
            'class2_pct': class_percentages[2],
            'class3_pct': class_percentages[3],
            'is_bg_heavy': is_bg_heavy
        })
        
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    # Filter for images with <= threshold background
    filtered_df = df[df['background_pct'] < threshold]

    # Print the filtered list
    print("\nImages/masks with <= {:.0f}% background:".format(threshold*100))
    for idx, row in filtered_df.iterrows():
        print(f"Image: {row['image_name']}, Mask: {row['mask_name']}")

    # Calculate and print statistics
    print("\nAnalysis Results:")
    print(f"Total images: {total_images}")
    print(f"Images with >={threshold*100}% background: {heavy_bg_count} ({heavy_bg_count/total_images*100:.2f}%)")
    print(f"Images with <{threshold*100}% background: {total_images - heavy_bg_count} ({(total_images - heavy_bg_count)/total_images*100:.2f}%)")
    
    print("\nBackground percentage statistics:")
    print(f"  Min: {df['background_pct'].min():.4f}")
    print(f"  Max: {df['background_pct'].max():.4f}")
    print(f"  Mean: {df['background_pct'].mean():.4f}")
    print(f"  Median: {df['background_pct'].median():.4f}")
    
    # Save results to CSV
    output_csv = os.path.join(data_dir, f"{mode}_cleaned_patches.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nAnalysis results saved to {output_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze glacier dataset for background-heavy images')
    parser.add_argument('--data_dir', type=str, required=True, help='Root data directory')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'], 
                        help='Dataset split to analyze')
    parser.add_argument('--threshold', type=float, default=0.8, 
                        help='Background percentage threshold (0-1)')
   
    args = parser.parse_args()
    
    # Run analysis
    analyze_dataset(
        data_dir=args.data_dir,
        mode=args.mode, 
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()