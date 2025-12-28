"""
Complete Pipeline Runner
Executes the entire multimodal regression pipeline
"""
import argparse
import pandas as pd
from pathlib import Path
import sys

from config import Config
from data_fetcher import SatelliteImageFetcher


def download_images(train_path, test_path):
    """Step 1: Download satellite images"""
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING SATELLITE IMAGES")
    print("="*70)
    
    fetcher = SatelliteImageFetcher()
    
    # Load data
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    
    # Download training images
    print("\n[1/2] Downloading training images...")
    train_with_images = fetcher.fetch_and_save_images(
        train_df,
        save_dir=Config.IMAGE_SAVE_DIR / 'train',
        delay=0.1
    )
    
    # Download test images
    print("\n[2/2] Downloading test images...")
    test_with_images = fetcher.fetch_and_save_images(
        test_df,
        save_dir=Config.IMAGE_SAVE_DIR / 'test',
        delay=0.1
    )
    
    # Save datasets with image paths
    output_dir = Config.DATA_DIR / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_output = output_dir / 'train_with_images.csv'
    test_output = output_dir / 'test_with_images.csv'
    
    train_with_images.to_csv(train_output, index=False)
    test_with_images.to_csv(test_output, index=False)
    
    print(f"\n✓ Saved data with image paths:")
    print(f"  - {train_output}")
    print(f"  - {test_output}")
    
    return train_with_images, test_with_images


def preprocess_data(train_df, test_df):
    """Step 2: Preprocess and engineer features"""
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    
    def engineer_features(df):
        """Create engineered features"""
        df = df.copy()
        
        # Age features
        current_year = 2024
        df['age'] = current_year - df['yr_built']
        df['years_since_renovation'] = current_year - df['yr_renovated']
        df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)
        
        # Size ratios
        df['living_lot_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1)
        df['above_ground_ratio'] = df['sqft_above'] / (df['sqft_living'] + 1)
        df['basement_ratio'] = df['sqft_basement'] / (df['sqft_living'] + 1)
        
        # Room ratios
        df['bath_bed_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1)
        df['rooms_per_sqft'] = (df['bedrooms'] + df['bathrooms']) / (df['sqft_living'] + 1)
        
        # Neighborhood
        df['living_vs_neighbors'] = df['sqft_living'] / (df['sqft_living15'] + 1)
        df['lot_vs_neighbors'] = df['sqft_lot'] / (df['sqft_lot15'] + 1)
        
        # Quality
        df['overall_quality'] = df['grade'] * df['condition']
        df['luxury_score'] = df['grade'] + df['view'] + df['waterfront']*2
        
        # Log transforms
        for col in ['sqft_living', 'sqft_lot', 'sqft_above']:
            if col in df.columns:
                df[f'log_{col}'] = np.log1p(df[col])
        
        return df
    
    import numpy as np
    
    train_engineered = engineer_features(train_df)
    test_engineered = engineer_features(test_df)
    
    print(f"\n✓ Created {train_engineered.shape[1] - train_df.shape[1]} new features")
    
    # Save processed data
    output_dir = Config.DATA_DIR / 'processed'
    train_output = output_dir / 'train_processed.csv'
    test_output = output_dir / 'test_processed.csv'
    
    train_engineered.to_csv(train_output, index=False)
    test_engineered.to_csv(test_output, index=False)
    
    print(f"\n✓ Saved processed data:")
    print(f"  - {train_output}")
    print(f"  - {test_output}")
    
    return train_engineered, test_engineered


def train_models():
    """Step 3: Train models"""
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    print("\nPlease run the following command to train models:")
    print("  jupyter notebook model_training.ipynb")
    print("\nOr use the command line training script:")
    print("  python train.py")


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Run Real Estate Multimodal Pipeline')
    parser.add_argument('--skip-images', action='store_true', 
                       help='Skip image downloading (if already done)')
    parser.add_argument('--train-path', type=str, default=Config.TRAIN_DATA_PATH,
                       help='Path to training data')
    parser.add_argument('--test-path', type=str, default=Config.TEST_DATA_PATH,
                       help='Path to test data')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("REAL ESTATE MULTIMODAL VALUATION PIPELINE")
    print("="*70)
    
    # Validate configuration
    try:
        Config.validate_api_config()
        print("\n✓ API configuration validated")
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        print("\nPlease:")
        print("1. Copy .env.template to .env")
        print("2. Add your API key to .env")
        print("3. Run this script again")
        sys.exit(1)
    
    Config.print_config()
    
    # Step 1: Download images
    if not args.skip_images:
        train_df, test_df = download_images(args.train_path, args.test_path)
    else:
        print("\n⊗ Skipping image download (--skip-images flag set)")
        processed_dir = Config.DATA_DIR / 'processed'
        train_df = pd.read_csv(processed_dir / 'train_with_images.csv')
        test_df = pd.read_csv(processed_dir / 'test_with_images.csv')
    
    # Step 2: Preprocess data
    train_processed, test_processed = preprocess_data(train_df, test_df)
    
    # Step 3: Instructions for training
    train_models()
    
    print("\n" + "="*70)
    print("PIPELINE SETUP COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review downloaded images in:", Config.IMAGE_SAVE_DIR)
    print("2. Check processed data in:", Config.DATA_DIR / 'processed')
    print("3. Open and run: model_training.ipynb")
    print("4. Generate predictions on test data")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()