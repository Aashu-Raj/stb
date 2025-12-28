"""
Satellite Image Fetcher Module
Downloads satellite images using coordinates from various APIs
"""
import requests
import time
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import pandas as pd

from config import Config


class SatelliteImageFetcher:
    """Fetch satellite images from various providers"""
    
    def __init__(self, provider: str = None):
        """
        Initialize the fetcher
        
        Args:
            provider: 'google', 'mapbox', or 'sentinel'
        """
        self.provider = provider or Config.SATELLITE_API_PROVIDER
        self.validate_credentials()
        
    def validate_credentials(self):
        """Validate API credentials"""
        Config.validate_api_config()
        
    def fetch_google_maps_image(
        self, 
        lat: float, 
        lon: float, 
        zoom: int = None,
        size: Tuple[int, int] = None
    ) -> Optional[Image.Image]:
        """
        Fetch satellite image from Google Maps Static API
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level (default from config)
            size: Image size tuple (width, height)
            
        Returns:
            PIL Image or None if failed
        """
        zoom = zoom or Config.IMAGE_ZOOM
        size = size or (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT)
        
        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            'center': f'{lat},{lon}',
            'zoom': zoom,
            'size': f'{size[0]}x{size[1]}',
            'maptype': 'satellite',
            'key': Config.GOOGLE_MAPS_API_KEY,
            'format': Config.IMAGE_FORMAT
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            return img
        except Exception as e:
            print(f"Error fetching Google Maps image for ({lat}, {lon}): {e}")
            return None
    
    def fetch_mapbox_image(
        self,
        lat: float,
        lon: float,
        zoom: int = None,
        size: Tuple[int, int] = None
    ) -> Optional[Image.Image]:
        """
        Fetch satellite image from Mapbox Static Images API
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level (default from config)
            size: Image size tuple (width, height)
            
        Returns:
            PIL Image or None if failed
        """
        zoom = zoom or Config.IMAGE_ZOOM
        size = size or (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT)
        
        # Mapbox URL format
        base_url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"
        url = f"{base_url}/{lon},{lat},{zoom}/{size[0]}x{size[1]}"
        
        params = {
            'access_token': Config.MAPBOX_ACCESS_TOKEN
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            return img
        except Exception as e:
            print(f"Error fetching Mapbox image for ({lat}, {lon}): {e}")
            return None
    
    def fetch_sentinel_image(
        self,
        lat: float,
        lon: float,
        zoom: int = None,
        size: Tuple[int, int] = None
    ) -> Optional[Image.Image]:
        """
        Fetch satellite image from Sentinel Hub
        Note: This is a simplified implementation
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level (default from config)
            size: Image size tuple (width, height)
            
        Returns:
            PIL Image or None if failed
        """
        # Note: Sentinel Hub requires OAuth authentication
        # This is a simplified version - full implementation would need proper OAuth flow
        print("Note: Sentinel Hub integration requires additional setup")
        print("Falling back to Google Maps API")
        return self.fetch_google_maps_image(lat, lon, zoom, size)
    
    def fetch_image(
        self,
        lat: float,
        lon: float,
        property_id: str = None
    ) -> Optional[Image.Image]:
        """
        Fetch image using configured provider
        
        Args:
            lat: Latitude
            lon: Longitude
            property_id: Optional property identifier for logging
            
        Returns:
            PIL Image or None if failed
        """
        if self.provider == 'google':
            return self.fetch_google_maps_image(lat, lon)
        elif self.provider == 'mapbox':
            return self.fetch_mapbox_image(lat, lon)
        elif self.provider == 'sentinel':
            return self.fetch_sentinel_image(lat, lon)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def fetch_and_save_images(
        self,
        df: pd.DataFrame,
        save_dir: Path = None,
        id_column: str = 'id',
        lat_column: str = 'lat',
        lon_column: str = 'long',
        delay: float = 0.1
    ) -> pd.DataFrame:
        """
        Fetch and save images for all properties in dataframe
        
        Args:
            df: DataFrame with coordinates
            save_dir: Directory to save images
            id_column: Column name for property ID
            lat_column: Column name for latitude
            lon_column: Column name for longitude
            delay: Delay between requests (seconds)
            
        Returns:
            DataFrame with added 'image_path' column
        """
        save_dir = save_dir or Config.IMAGE_SAVE_DIR
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        
        print(f"\nFetching {len(df)} satellite images using {self.provider} API...")
        print(f"Saving to: {save_dir}")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
            prop_id = row[id_column]
            lat = row[lat_column]
            lon = row[lon_column]
            
            # Define save path
            image_filename = f"{prop_id}.{Config.IMAGE_FORMAT}"
            image_path = save_dir / image_filename
            
            # Skip if already exists
            if image_path.exists():
                image_paths.append(str(image_path))
                continue
            
            # Fetch image
            img = self.fetch_image(lat, lon, property_id=prop_id)
            
            if img is not None:
                # Save image
                img.save(image_path)
                image_paths.append(str(image_path))
            else:
                # Handle failure
                image_paths.append(None)
                print(f"Failed to fetch image for property {prop_id}")
            
            # Rate limiting
            time.sleep(delay)
        
        # Add image paths to dataframe
        df_with_images = df.copy()
        df_with_images['image_path'] = image_paths
        
        # Report statistics
        success_count = df_with_images['image_path'].notna().sum()
        print(f"\n✓ Successfully downloaded {success_count}/{len(df)} images")
        
        return df_with_images


def main():
    """Example usage"""
    import sys
    
    # Check if data file provided
    if len(sys.argv) < 2:
        print("Usage: python data_fetcher.py <path_to_data.xlsx>")
        print("Example: python data_fetcher.py data/train(1).xlsx")
        return
    
    data_path = sys.argv[1]
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_excel(data_path)
    
    # Initialize fetcher
    fetcher = SatelliteImageFetcher()
    
    # Fetch images
    df_with_images = fetcher.fetch_and_save_images(df)
    
    # Save updated dataframe
    output_path = Path(data_path).parent / f"{Path(data_path).stem}_with_images.csv"
    df_with_images.to_csv(output_path, index=False)
    print(f"\n✓ Saved updated data to {output_path}")


if __name__ == "__main__":
    main()