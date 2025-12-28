"""
Multimodal Model Architecture
Combines CNN for image features with Neural Network for tabular data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageFeatureExtractor(nn.Module):
    """
    CNN-based feature extractor for satellite images
    Uses pretrained ResNet50 as backbone
    """
    def __init__(self, embedding_dim=512, pretrained=True):
        super(ImageFeatureExtractor, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add custom embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        Args:
            x: Image tensor of shape (batch_size, 3, H, W)
        Returns:
            Image embeddings of shape (batch_size, embedding_dim)
        """
        # Extract features
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Generate embeddings
        x = self.embedding(x)
        
        return x


class TabularNetwork(nn.Module):
    """
    Neural network for processing tabular features
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(TabularNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x):
        """
        Args:
            x: Tabular features (batch_size, input_dim)
        Returns:
            Processed features (batch_size, output_dim)
        """
        return self.network(x)


class MultimodalFusionModel(nn.Module):
    """
    Multimodal model that fuses image and tabular features
    
    Fusion Strategies:
    - 'early': Concatenate features early and process together
    - 'late': Process separately and fuse at final layer
    - 'attention': Use attention mechanism for fusion
    """
    def __init__(
        self, 
        tabular_input_dim,
        image_embedding_dim=512,
        tabular_hidden_dims=[256, 128, 64],
        fusion_strategy='late',
        dropout=0.3
    ):
        super(MultimodalFusionModel, self).__init__()
        
        self.fusion_strategy = fusion_strategy
        
        # Image feature extractor
        self.image_extractor = ImageFeatureExtractor(
            embedding_dim=image_embedding_dim
        )
        
        # Tabular network
        self.tabular_network = TabularNetwork(
            input_dim=tabular_input_dim,
            hidden_dims=tabular_hidden_dims,
            dropout=dropout
        )
        
        # Fusion layer
        if fusion_strategy == 'early':
            fusion_input_dim = image_embedding_dim + tabular_input_dim
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )
        elif fusion_strategy == 'late':
            fusion_input_dim = image_embedding_dim + self.tabular_network.output_dim
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
        elif fusion_strategy == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=image_embedding_dim,
                num_heads=8,
                dropout=dropout
            )
            fusion_input_dim = image_embedding_dim + self.tabular_network.output_dim
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )
        
    def forward(self, images, tabular_features):
        """
        Args:
            images: Image tensor (batch_size, 3, H, W)
            tabular_features: Tabular features (batch_size, input_dim)
        Returns:
            Predicted price (batch_size, 1)
        """
        # Extract image features
        image_features = self.image_extractor(images)
        
        # Process tabular features
        if self.fusion_strategy == 'early':
            # Concatenate raw features
            combined = torch.cat([image_features, tabular_features], dim=1)
            output = self.fusion(combined)
        else:
            # Process tabular features
            tabular_processed = self.tabular_network(tabular_features)
            
            if self.fusion_strategy == 'attention':
                # Apply attention
                image_features_att = image_features.unsqueeze(0)
                tabular_features_att = tabular_processed.unsqueeze(0)
                attended_features, _ = self.attention(
                    image_features_att,
                    tabular_features_att,
                    tabular_features_att
                )
                attended_features = attended_features.squeeze(0)
                
                # Concatenate attended image features with tabular
                combined = torch.cat([attended_features, tabular_processed], dim=1)
            else:
                # Late fusion: concatenate processed features
                combined = torch.cat([image_features, tabular_processed], dim=1)
            
            output = self.fusion(combined)
        
        return output


class TabularOnlyModel(nn.Module):
    """
    Baseline model using only tabular features
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(TabularOnlyModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class ImageOnlyModel(nn.Module):
    """
    Model using only image features (for ablation study)
    """
    def __init__(self, embedding_dim=512, dropout=0.3):
        super(ImageOnlyModel, self).__init__()
        
        self.image_extractor = ImageFeatureExtractor(embedding_dim=embedding_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, images):
        features = self.image_extractor(images)
        return self.regressor(features)


def get_model(model_type='multimodal', **kwargs):
    """
    Factory function to get model by type
    
    Args:
        model_type: 'multimodal', 'tabular_only', or 'image_only'
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Model instance
    """
    if model_type == 'multimodal':
        return MultimodalFusionModel(**kwargs)
    elif model_type == 'tabular_only':
        return TabularOnlyModel(**kwargs)
    elif model_type == 'image_only':
        return ImageOnlyModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model architectures
    batch_size = 4
    tabular_dim = 20
    
    print("Testing Multimodal Model...")
    model = MultimodalFusionModel(
        tabular_input_dim=tabular_dim,
        fusion_strategy='late'
    )
    
    # Create dummy data
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_tabular = torch.randn(batch_size, tabular_dim)
    
    # Forward pass
    output = model(dummy_images, dummy_tabular)
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n✓ Model architecture test passed!")