"""
Model Explainability Module
Implements Grad-CAM for visualizing important image regions
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Visualizes which parts of the image influence predictions
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: The multimodal model
            target_layer: The convolutional layer to visualize
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hook_layers()
        
    def hook_layers(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, image, tabular, denormalize=True):
        """
        Generate Grad-CAM heatmap
        
        Args:
            image: Input image tensor (1, 3, H, W)
            tabular: Tabular features (1, num_features)
            denormalize: Whether to denormalize the image
            
        Returns:
            cam: Heatmap as numpy array
            prediction: Model prediction
        """
        self.model.eval()
        
        # Forward pass
        image.requires_grad = True
        output = self.model(image, tabular)
        prediction = output.item()
        
        # Backward pass
        self.model.zero_grad()
        output.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H', W')
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to original image size
        cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
        
        return cam, prediction
    
    def visualize(
        self,
        image,
        tabular,
        cam,
        prediction,
        actual_price=None,
        save_path=None
    ):
        """
        Visualize the Grad-CAM heatmap overlaid on image
        
        Args:
            image: Original image tensor (1, 3, H, W)
            tabular: Tabular features
            cam: Grad-CAM heatmap
            prediction: Predicted price
            actual_price: Actual price (optional)
            save_path: Path to save visualization
        """
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image.squeeze(0) * std + mean
        image = image.clamp(0, 1)
        image_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap / 255.0
        
        # Overlay heatmap on image
        overlayed = 0.5 * image_np + 0.5 * heatmap
        overlayed = np.clip(overlayed, 0, 1)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Satellite Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlayed
        axes[2].imshow(overlayed)
        title = f'Predicted: ${prediction:,.0f}'
        if actual_price is not None:
            error = abs(prediction - actual_price)
            error_pct = 100 * error / actual_price
            title += f'\nActual: ${actual_price:,.0f}\nError: ${error:,.0f} ({error_pct:.1f}%)'
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def analyze_multiple_samples(
    model,
    dataset,
    indices,
    target_layer,
    device='cuda',
    save_dir=None
):
    """
    Analyze multiple samples with Grad-CAM
    
    Args:
        model: Trained multimodal model
        dataset: Dataset to sample from
        indices: List of sample indices to analyze
        target_layer: Target conv layer for Grad-CAM
        device: Device to use
        save_dir: Directory to save visualizations
    """
    gradcam = GradCAM(model, target_layer)
    
    model.eval()
    
    for idx in indices:
        print(f"\nAnalyzing sample {idx}...")
        
        # Get sample
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        tabular = sample['tabular'].unsqueeze(0).to(device)
        
        if 'target' in sample:
            actual_price = sample['target'].item()
        else:
            actual_price = None
        
        # Generate CAM
        cam, prediction = gradcam.generate_cam(image, tabular)
        
        # Visualize
        save_path = None
        if save_dir:
            save_path = save_dir / f'gradcam_sample_{idx}.png'
        
        gradcam.visualize(
            image=image,
            tabular=tabular,
            cam=cam,
            prediction=prediction,
            actual_price=actual_price,
            save_path=save_path
        )


def find_interesting_samples(model, dataset, device='cuda', num_samples=10):
    """
    Find interesting samples for Grad-CAM visualization
    Returns indices of samples with:
    - High prediction error
    - Low prediction error
    - Expensive properties
    - Cheap properties
    
    Args:
        model: Trained model
        dataset: Dataset
        device: Device
        num_samples: Number of samples to return
        
    Returns:
        Dictionary of sample indices by category
    """
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            tabular = sample['tabular'].unsqueeze(0).to(device)
            
            pred = model(image, tabular).item()
            predictions.append(pred)
            
            if 'target' in sample:
                actuals.append(sample['target'].item())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate errors
    errors = np.abs(predictions - actuals)
    
    # Find interesting samples
    interesting = {
        'high_error': np.argsort(errors)[-num_samples:].tolist(),
        'low_error': np.argsort(errors)[:num_samples].tolist(),
        'expensive': np.argsort(predictions)[-num_samples:].tolist(),
        'cheap': np.argsort(predictions)[:num_samples].tolist()
    }
    
    return interesting


def create_feature_importance_map(
    model,
    dataset,
    sample_indices,
    device='cuda'
):
    """
    Create aggregated feature importance visualization
    Shows which visual features are consistently important
    
    Args:
        model: Trained model
        dataset: Dataset
        sample_indices: Indices to analyze
        device: Device
        
    Returns:
        Average heatmap across samples
    """
    # Get target layer
    target_layer = model.image_extractor.features[-1]
    gradcam = GradCAM(model, target_layer)
    
    all_cams = []
    
    model.eval()
    
    for idx in sample_indices:
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        tabular = sample['tabular'].unsqueeze(0).to(device)
        
        cam, _ = gradcam.generate_cam(image, tabular)
        all_cams.append(cam)
    
    # Average CAMs
    avg_cam = np.mean(all_cams, axis=0)
    
    # Visualize
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_cam, cmap='jet')
    plt.colorbar(label='Average Importance')
    plt.title(f'Average Feature Importance\n({len(sample_indices)} samples)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return avg_cam


if __name__ == "__main__":
    print("Explainability module loaded")
    print("Use analyze_multiple_samples() to visualize Grad-CAM")