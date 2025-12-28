"""
Prediction Script
Load trained model and generate predictions on test data
"""
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

from config import Config
from model_architecture import MultimodalFusionModel
from dataset import MultimodalDataset, get_val_transforms
from trainer import predict


def load_model(model_path, model_type='multimodal', fusion_strategy='late', num_features=20, device='cuda'):
    """Load trained model from checkpoint"""
    if model_type == 'multimodal':
        model = MultimodalFusionModel(
            tabular_input_dim=num_features,
            image_embedding_dim=512,
            tabular_hidden_dims=[256, 128, 64],
            fusion_strategy=fusion_strategy,
            dropout=0.3
        )
    else:
        from model_architecture import TabularOnlyModel
        model = TabularOnlyModel(
            input_dim=num_features,
            hidden_dims=[256, 128, 64],
            dropout=0.3
        )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {model_path}")
    
    return model


def prepare_test_data(test_path, scaler_path=None):
    """Prepare test data for prediction"""
    # Load test data
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded test data: {test_df.shape}")
    
    # Define features
    tabular_features = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
        'floors', 'waterfront', 'view', 'condition', 'grade',
        'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
        'sqft_living15', 'sqft_lot15', 'lat', 'long',
        'age', 'years_since_renovation', 'is_renovated',
        'living_lot_ratio', 'above_ground_ratio', 'basement_ratio',
        'bath_bed_ratio', 'rooms_per_sqft',
        'living_vs_neighbors', 'lot_vs_neighbors',
        'overall_quality', 'luxury_score',
        'log_sqft_living', 'log_sqft_lot', 'log_sqft_above'
    ]
    
    # Filter available features
    available_features = [f for f in tabular_features if f in test_df.columns]
    print(f"Using {len(available_features)} features")
    
    # Handle missing values
    for col in available_features:
        if test_df[col].isnull().any():
            median_val = test_df[col].median()
            test_df[col].fillna(median_val, inplace=True)
    
    # Scale features
    if scaler_path and Path(scaler_path).exists():
        scaler = joblib.load(scaler_path)
        print(f"✓ Loaded scaler from {scaler_path}")
    else:
        print("⚠ No scaler provided, using StandardScaler on test data")
        scaler = StandardScaler()
        scaler.fit(test_df[available_features])
    
    test_df[available_features] = scaler.transform(test_df[available_features])
    
    return test_df, available_features


def main():
    parser = argparse.ArgumentParser(description='Generate predictions on test data')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test-path', type=str, 
                       default='data/processed/test_processed.csv',
                       help='Path to test data')
    parser.add_argument('--output-path', type=str, default='predictions.csv',
                       help='Path to save predictions')
    parser.add_argument('--model-type', type=str, default='multimodal',
                       choices=['multimodal', 'tabular'],
                       help='Type of model')
    parser.add_argument('--fusion-strategy', type=str, default='late',
                       choices=['early', 'late', 'attention'],
                       help='Fusion strategy (for multimodal only)')
    parser.add_argument('--scaler-path', type=str, default=None,
                       help='Path to saved scaler')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for prediction')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Prepare test data
    print("\nPreparing test data...")
    test_df, features = prepare_test_data(args.test_path, args.scaler_path)
    
    # Load model
    print("\nLoading model...")
    model = load_model(
        args.model_path,
        model_type=args.model_type,
        fusion_strategy=args.fusion_strategy,
        num_features=len(features),
        device=device
    )
    
    # Create dataset and dataloader
    print("\nCreating dataset...")
    if args.model_type == 'multimodal':
        test_dataset = MultimodalDataset(
            df=test_df,
            tabular_features=features,
            transform=get_val_transforms(),
            is_test=True
        )
    else:
        from dataset import TabularOnlyDataset
        test_dataset = TabularOnlyDataset(
            df=test_df,
            tabular_features=features,
            is_test=True
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Created dataset with {len(test_dataset)} samples")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions, ids = predict(model, test_loader, device=device)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': ids,
        'predicted_price': predictions
    })
    
    submission_df = submission_df.sort_values('id')
    
    # Save predictions
    submission_df.to_csv(args.output_path, index=False)
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    print(f"\n✓ Predictions saved to: {args.output_path}")
    print(f"\nPrediction Statistics:")
    print(submission_df['predicted_price'].describe())
    print("\n" + "="*70)


if __name__ == "__main__":
    main()