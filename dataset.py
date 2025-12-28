"""
PyTorch Dataset Classes for Multimodal Data
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal real estate data (images + tabular)
    """
    def __init__(
        self, 
        df, 
        tabular_features,
        target_column='price',
        transform=None,
        is_test=False
    ):
        """
        Args:
            df: DataFrame with image paths and features
            tabular_features: List of column names for tabular features
            target_column: Name of target column (price)
            transform: Image transformations
            is_test: Whether this is test data (no target)
        """
        self.df = df.reset_index(drop=True)
        self.tabular_features = tabular_features
        self.target_column = target_column
        self.is_test = is_test
        
        # Filter to only rows with valid images
        self.df = self.df[self.df['image_path'].notna()].reset_index(drop=True)
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Prepare tabular features
        self.tabular_data = self.df[tabular_features].values.astype(np.float32)
        
        # Prepare targets (if not test)
        if not is_test:
            self.targets = self.df[target_column].values.astype(np.float32)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.df.iloc[idx]['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # If image fails to load, return a blank image
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # Get tabular features
        tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        
        if self.is_test:
            return {
                'image': image,
                'tabular': tabular,
                'id': self.df.iloc[idx]['id']
            }
        else:
            # Get target
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            
            return {
                'image': image,
                'tabular': tabular,
                'target': target
            }


class TabularOnlyDataset(Dataset):
    """
    Dataset for tabular-only baseline
    """
    def __init__(self, df, tabular_features, target_column='price', is_test=False):
        self.df = df.reset_index(drop=True)
        self.tabular_features = tabular_features
        self.target_column = target_column
        self.is_test = is_test
        
        # Prepare features
        self.features = self.df[tabular_features].values.astype(np.float32)
        
        # Prepare targets
        if not is_test:
            self.targets = self.df[target_column].values.astype(np.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        
        if self.is_test:
            return {
                'tabular': features,
                'id': self.df.iloc[idx]['id']
            }
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return {
                'tabular': features,
                'target': target
            }


class ImageOnlyDataset(Dataset):
    """
    Dataset for image-only model (ablation study)
    """
    def __init__(self, df, target_column='price', transform=None, is_test=False):
        self.df = df[df['image_path'].notna()].reset_index(drop=True)
        self.target_column = target_column
        self.is_test = is_test
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Prepare targets
        if not is_test:
            self.targets = self.df[target_column].values.astype(np.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.df.iloc[idx]['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        if self.is_test:
            return {
                'image': image,
                'id': self.df.iloc[idx]['id']
            }
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return {
                'image': image,
                'target': target
            }


def get_train_transforms():
    """Get training image transformations with augmentation"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms():
    """Get validation image transformations (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


if __name__ == "__main__":
    # Test dataset
    from config import Config
    
    # Load sample data
    df = pd.read_csv(Config.DATA_DIR / 'processed' / 'train_processed.csv')
    
    # Define features
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'lat', 'long']
    
    # Create dataset
    dataset = MultimodalDataset(
        df=df,
        tabular_features=features,
        transform=get_train_transforms()
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample data:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Tabular shape: {sample['tabular'].shape}")
    print(f"  Target: {sample['target']:.2f}")
    
    print("\n✓ Dataset test passed!")