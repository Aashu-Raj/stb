"""
Configuration Module
Loads environment variables and provides centralized configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration class"""
    
    # ==================== API CONFIGURATION ====================
    SATELLITE_API_PROVIDER = os.getenv('SATELLITE_API_PROVIDER', 'google')
    
    # Google Maps API
    GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')
    
    # Mapbox API
    MAPBOX_ACCESS_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN', '')
    
    # Sentinel Hub API
    SENTINEL_HUB_CLIENT_ID = os.getenv('SENTINEL_HUB_CLIENT_ID', '')
    SENTINEL_HUB_CLIENT_SECRET = os.getenv('SENTINEL_HUB_CLIENT_SECRET', '')
    
    # ==================== IMAGE CONFIGURATION ====================
    IMAGE_WIDTH = int(os.getenv('IMAGE_WIDTH', 640))
    IMAGE_HEIGHT = int(os.getenv('IMAGE_HEIGHT', 640))
    IMAGE_ZOOM = int(os.getenv('IMAGE_ZOOM', 17))
    IMAGE_FORMAT = os.getenv('IMAGE_FORMAT', 'png')
    
    # ==================== DATA PATHS ====================
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    TRAIN_DATA_PATH = os.getenv('TRAIN_DATA_PATH', 'data/train(1).xlsx')
    TEST_DATA_PATH = os.getenv('TEST_DATA_PATH', 'data/test2.xlsx')
    IMAGE_SAVE_DIR = Path(os.getenv('IMAGE_SAVE_DIR', 'data/satellite_images'))
    MODEL_SAVE_DIR = Path(os.getenv('MODEL_SAVE_DIR', 'models'))
    
    # Create directories if they don't exist
    IMAGE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # ==================== MODEL CONFIGURATION ====================
    RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
    TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 16))
    EPOCHS = int(os.getenv('EPOCHS', 50))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
    
    # ==================== FEATURE CONFIGURATION ====================
    TABULAR_FEATURES = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
        'floors', 'waterfront', 'view', 'condition', 'grade',
        'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
        'sqft_living15', 'sqft_lot15', 'lat', 'long'
    ]
    
    TARGET = 'price'
    
    @classmethod
    def validate_api_config(cls):
        """Validate that required API keys are present"""
        provider = cls.SATELLITE_API_PROVIDER.lower()
        
        if provider == 'google':
            if not cls.GOOGLE_MAPS_API_KEY:
                raise ValueError("GOOGLE_MAPS_API_KEY not set in .env file")
            return True
        elif provider == 'mapbox':
            if not cls.MAPBOX_ACCESS_TOKEN:
                raise ValueError("MAPBOX_ACCESS_TOKEN not set in .env file")
            return True
        elif provider == 'sentinel':
            if not cls.SENTINEL_HUB_CLIENT_ID or not cls.SENTINEL_HUB_CLIENT_SECRET:
                raise ValueError("Sentinel Hub credentials not set in .env file")
            return True
        else:
            raise ValueError(f"Invalid SATELLITE_API_PROVIDER: {provider}")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 70)
        print("CURRENT CONFIGURATION")
        print("=" * 70)
        print(f"API Provider: {cls.SATELLITE_API_PROVIDER}")
        print(f"Image Size: {cls.IMAGE_WIDTH}x{cls.IMAGE_HEIGHT}")
        print(f"Image Zoom: {cls.IMAGE_ZOOM}")
        print(f"Train Data: {cls.TRAIN_DATA_PATH}")
        print(f"Test Data: {cls.TEST_DATA_PATH}")
        print(f"Image Directory: {cls.IMAGE_SAVE_DIR}")
        print(f"Model Directory: {cls.MODEL_SAVE_DIR}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print("=" * 70)

if __name__ == "__main__":
    Config.print_config()
    try:
        Config.validate_api_config()
        print("\n✓ API Configuration Valid")
    except ValueError as e:
        print(f"\n✗ Configuration Error: {e}")