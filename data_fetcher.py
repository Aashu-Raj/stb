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
import numpy as np

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
    
    def get_sentinel_oauth_token(self) -> Optional[str]:
        """
        Get OAuth token for Sentinel Hub API
        
        Returns:
            Access token or None if failed
        """
        token_url = "https://services.sentinel-hub.com/oauth/token"
        
        data = {
            'grant_type': 'client_credentials',
            'client_id': Config.SENTINEL_HUB_CLIENT_ID,
            'client_secret': Config.SENTINEL_HUB_CLIENT_SECRET
        }
        
        try:
            response = requests.post(token_url, data=data, timeout=30)
            response.raise_for_status()
            token = response.json()['access_token']
            return token
        except Exception as e:
            print(f"Error getting Sentinel Hub OAuth token: {e}")
            return None
    
    def calculate_bbox(self, lat: float, lon: float, zoom: int, size: Tuple[int, int]) -> list:
        """
        Calculate bounding box for Sentinel Hub request
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level (not directly used, but kept for consistency)
            size: Image size
            
        Returns:
            Bounding box [min_lon, min_lat, max_lon, max_lat]
        """
        # Approximate meters per pixel at equator for different zoom levels
        # This is a rough approximation
        meters_per_pixel = 156543.03392 * np.cos(lat * np.pi / 180) / (2 ** zoom)
        
        # Calculate width and height in meters
        width_m = size[0] * meters_per_pixel
        height_m = size[1] * meters_per_pixel
        
        # Convert meters to degrees (approximate)
        # 1 degree latitude ≈ 111,320 meters
        # 1 degree longitude varies by latitude
        lat_offset = height_m / 111320 / 2
        lon_offset = width_m / (111320 * np.cos(lat * np.pi / 180)) / 2
        
        bbox = [
            lon - lon_offset,  # min_lon
            lat - lat_offset,  # min_lat
            lon + lon_offset,  # max_lon
            lat + lat_offset   # max_lat
        ]
        
        return bbox
    
    def fetch_sentinel_image(
        self,
        lat: float,
        lon: float,
        zoom: int = None,
        size: Tuple[int, int] = None
    ) -> Optional[Image.Image]:
        """
        Fetch satellite image from Sentinel Hub using Sentinel-2 L2A data
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level (default from config)
            size: Image size tuple (width, height)
            
        Returns:
            PIL Image or None if failed
        """
        import numpy as np
        
        zoom = zoom or Config.IMAGE_ZOOM
        size = size or (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT)
        
        # Get OAuth token
        token = self.get_sentinel_oauth_token()
        if not token:
            print("Failed to get Sentinel Hub OAuth token")
            return None
        
        # Calculate bounding box
        bbox = self.calculate_bbox(lat, lon, zoom, size)
        
        # Sentinel Hub Process API endpoint
        url = "https://services.sentinel-hub.com/api/v1/process"
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'image/png'
        }
        
        # Request payload for true color RGB image
        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {
                        "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                    }
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": "2023-01-01T00:00:00Z",
                                "to": "2024-12-31T23:59:59Z"
                            },
                            "maxCloudCoverage": 30
                        }
                    }
                ]
            },
            "output": {
                "width": size[0],
                "height": size[1],
                "responses": [
                    {
                        "identifier": "default",
                        "format": {
                            "type": "image/png"
                        }
                    }
                ]
            },
            "evalscript": """
                //VERSION=3
                function setup() {
                    return {
                        input: ["B04", "B03", "B02"],
                        output: {
                            bands: 3,
                            sampleType: "AUTO"
                        }
                    };
                }
                
                function evaluatePixel(sample) {
                    // True color RGB with slight enhancement
                    return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
                }
            """
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            # Load image from response
            img = Image.open(BytesIO(response.content))
            return img
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                print(f"Bad request for ({lat}, {lon}): {e.response.text}")
            elif e.response.status_code == 401:
                print(f"Authentication error - check your credentials")
            elif e.response.status_code == 404:
                print(f"No Sentinel data available for location ({lat}, {lon})")
            else:
                print(f"HTTP error fetching Sentinel image for ({lat}, {lon}): {e}")
            return None
        except Exception as e:
            print(f"Error fetching Sentinel image for ({lat}, {lon}): {e}")
            return None
    
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