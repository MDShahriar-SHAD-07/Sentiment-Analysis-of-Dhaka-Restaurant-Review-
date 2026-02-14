"""
Data Loader for FastAPI Application
===================================

Loads processed data from notebook and initializes vector store
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional
import pickle
import json

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DataLoader:
    """Load and manage processed restaurant review data"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.restaurant_index: Optional[dict] = None
        
    def load_processed_data(self) -> pd.DataFrame:
        """
        Load processed data from CSV/pickle
        
        Returns:
            DataFrame with processed reviews
        """
        try:
            # Try loading from CSV first
            if settings.PROCESSED_DATA_PATH.exists():
                logger.info(f"Loading data from {settings.PROCESSED_DATA_PATH}")
                self.df = pd.read_csv(settings.PROCESSED_DATA_PATH)
                logger.info(f"✅ Loaded {len(self.df)} reviews")
                return self.df
            
            # Try pickle as fallback
            pickle_path = settings.DATA_DIR / "processed_reviews.pkl"
            if pickle_path.exists():
                logger.info(f"Loading data from {pickle_path}")
                self.df = pd.read_pickle(pickle_path)
                logger.info(f"✅ Loaded {len(self.df)} reviews")
                return self.df
            
            raise FileNotFoundError(
                f"No processed data found. Please run the notebook first to generate "
                f"processed data at {settings.PROCESSED_DATA_PATH}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def build_restaurant_index(self) -> dict:
        """
        Build restaurant name index for fast lookup
        
        Returns:
            Dictionary mapping normalized names to original names
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_processed_data() first.")
        
        logger.info("Building restaurant index...")
        
        index = {}
        for name in self.df['business_name'].unique():
            normalized = name.lower().strip()
            index[normalized] = name
            
            # Add partial matches for better search
            words = normalized.split()
            for word in words:
                if len(word) > 3:  # Avoid short words
                    if word not in index:  # Don't overwrite existing entries
                        index[word] = name
        
        self.restaurant_index = index
        logger.info(f"✅ Built index with {len(index)} entries")
        
        return index
    
    def get_restaurant_stats(self) -> dict:
        """
        Get basic statistics about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        return {
            "total_reviews": len(self.df),
            "total_restaurants": self.df['business_name'].nunique(),
            "avg_reviews_per_restaurant": len(self.df) / self.df['business_name'].nunique(),
            "sentiment_distribution": self.df['overall_sentiment'].value_counts().to_dict(),
            "avg_rating": float(self.df['review_rating'].mean()),
            "date_range": {
                "earliest": str(self.df['review_date'].min()) if 'review_date' in self.df.columns else "N/A",
                "latest": str(self.df['review_date'].max()) if 'review_date' in self.df.columns else "N/A"
            }
        }
    
    def export_restaurant_list(self, output_path: Optional[Path] = None) -> dict:
        """
        Export list of all restaurants with basic info
        
        Args:
            output_path: Optional path to save JSON file
            
        Returns:
            Dictionary with restaurant list
        """
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        restaurants = []
        
        for name in sorted(self.df['business_name'].unique()):
            rest_df = self.df[self.df['business_name'] == name]
            
            restaurants.append({
                "name": name,
                "review_count": len(rest_df),
                "average_rating": float(rest_df['review_rating'].mean()),
                "sentiment_distribution": rest_df['overall_sentiment'].value_counts().to_dict()
            })
        
        result = {
            "total": len(restaurants),
            "restaurants": restaurants
        }
        
        # Save to file if path provided
        if output_path:
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Exported restaurant list to {output_path}")
        
        return result


def initialize_data() -> Tuple[pd.DataFrame, dict]:
    """
    Initialize data loading on application startup
    
    Returns:
        Tuple of (DataFrame, restaurant_index)
    """
    loader = DataLoader()
    df = loader.load_processed_data()
    restaurant_index = loader.build_restaurant_index()
    
    # Log statistics
    stats = loader.get_restaurant_stats()
    logger.info(f"📊 Dataset Statistics:")
    logger.info(f"  - Total Reviews: {stats['total_reviews']}")
    logger.info(f"  - Total Restaurants: {stats['total_restaurants']}")
    logger.info(f"  - Average Rating: {stats['avg_rating']:.2f}")
    
    return df, restaurant_index
