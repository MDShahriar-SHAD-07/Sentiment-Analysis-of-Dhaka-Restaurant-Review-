"""
Pydantic Models for FastAPI Request/Response Validation
=======================================================
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class AspectType(str, Enum):
    """Available aspect types"""
    FOOD = "food"
    SERVICE = "service"
    PRICE = "price"
    AMBIENCE = "ambience"
    CLEANLINESS = "cleanliness"


class SentimentType(str, Enum):
    """Sentiment types"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# ============================================
# Chat Endpoint Models
# ============================================

class ChatRequest(BaseModel):
    """Request model for /chat endpoint"""
    message: str = Field(..., min_length=1, max_length=500, 
                        description="User's question or message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Which restaurant has the best food?",
                "session_id": "user_123"
            }
        }


class SourceDocument(BaseModel):
    """Source document from vector store"""
    content: str = Field(..., description="Review text")
    restaurant: str = Field(..., description="Restaurant name")
    rating: float = Field(..., description="Review rating")
    sentiment: str = Field(..., description="Overall sentiment")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Amazing food quality and great service...",
                "restaurant": "Izumi Japanese Kitchen",
                "rating": 5.0,
                "sentiment": "positive"
            }
        }


class ChatResponse(BaseModel):
    """Response model for /chat endpoint"""
    response: str = Field(..., description="Bot's response")
    sources: List[SourceDocument] = Field(default_factory=list, 
                                         description="Source reviews used")
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                    description="Additional metadata")
    session_id: Optional[str] = Field(None, description="Session ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Based on 837 food-related reviews, Izumi Japanese Kitchen has the highest positive sentiment...",
                "sources": [
                    {
                        "content": "Amazing sushi quality...",
                        "restaurant": "Izumi Japanese Kitchen",
                        "rating": 5.0,
                        "sentiment": "positive"
                    }
                ],
                "metadata": {
                    "intent": "recommendation",
                    "aspects": ["food"],
                    "confidence": 0.85
                },
                "session_id": "user_123"
            }
        }


# ============================================
# Recommendation Endpoint Models
# ============================================

class RecommendationRequest(BaseModel):
    """Request model for /recommend endpoint"""
    aspect: Optional[AspectType] = Field(None, description="Specific aspect to filter by")
    min_reviews: int = Field(3, ge=1, le=100, 
                            description="Minimum number of reviews required")
    sentiment_preference: SentimentType = Field(SentimentType.POSITIVE,
                                               description="Preferred sentiment")
    limit: int = Field(5, ge=1, le=20, description="Maximum number of recommendations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "aspect": "food",
                "min_reviews": 5,
                "sentiment_preference": "positive",
                "limit": 5
            }
        }


class RestaurantRecommendation(BaseModel):
    """Single restaurant recommendation"""
    name: str = Field(..., description="Restaurant name")
    score: float = Field(..., ge=0, le=100, description="Recommendation score (0-100)")
    review_count: int = Field(..., description="Number of reviews")
    average_rating: float = Field(..., ge=0, le=5, description="Average rating")
    sentiment_distribution: Dict[str, int] = Field(..., 
                                                   description="Sentiment breakdown")
    aspect_scores: Optional[Dict[str, float]] = Field(None,
                                                      description="Aspect-wise scores")
    strengths: List[str] = Field(default_factory=list, description="Key strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Key weaknesses")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Izumi Japanese Kitchen",
                "score": 87.5,
                "review_count": 15,
                "average_rating": 4.6,
                "sentiment_distribution": {
                    "positive": 13,
                    "negative": 1,
                    "neutral": 1
                },
                "aspect_scores": {
                    "food": 92.0,
                    "service": 85.0,
                    "ambience": 88.0
                },
                "strengths": ["Excellent food", "Great ambience"],
                "weaknesses": []
            }
        }


class RecommendationResponse(BaseModel):
    """Response model for /recommend endpoint"""
    recommendations: List[RestaurantRecommendation] = Field(...,
                                                            description="List of recommendations")
    total_analyzed: int = Field(..., description="Total restaurants analyzed")
    filters_applied: Dict[str, Any] = Field(..., description="Filters used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "recommendations": [
                    {
                        "name": "Izumi Japanese Kitchen",
                        "score": 87.5,
                        "review_count": 15,
                        "average_rating": 4.6,
                        "sentiment_distribution": {"positive": 13, "negative": 1, "neutral": 1},
                        "aspect_scores": {"food": 92.0, "service": 85.0},
                        "strengths": ["Excellent food"],
                        "weaknesses": []
                    }
                ],
                "total_analyzed": 126,
                "filters_applied": {
                    "aspect": "food",
                    "min_reviews": 5
                }
            }
        }


# ============================================
# Analysis Endpoint Models
# ============================================

class AspectAnalysis(BaseModel):
    """Aspect-level analysis"""
    mentioned_count: int = Field(..., description="Number of times mentioned")
    positive_rate: float = Field(..., ge=0, le=1, description="Positive sentiment rate")
    negative_rate: float = Field(..., ge=0, le=1, description="Negative sentiment rate")
    neutral_rate: float = Field(..., ge=0, le=1, description="Neutral sentiment rate")


class AnalysisResponse(BaseModel):
    """Response model for /analyze/{restaurant_name} endpoint"""
    restaurant_name: str = Field(..., description="Restaurant name")
    total_reviews: int = Field(..., description="Total number of reviews")
    average_rating: float = Field(..., ge=0, le=5, description="Average rating")
    sentiment_distribution: Dict[str, int] = Field(..., 
                                                   description="Overall sentiment breakdown")
    aspect_analysis: Dict[str, AspectAnalysis] = Field(...,
                                                       description="Aspect-wise analysis")
    conflict_rate: float = Field(..., ge=0, le=1,
                                 description="Rate of conflicting reviews")
    strengths: List[str] = Field(default_factory=list, description="Key strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Key weaknesses")
    risk_factors: List[str] = Field(default_factory=list, 
                                   description="Potential risk factors")
    recommendation_score: float = Field(..., ge=0, le=100,
                                       description="Overall recommendation score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "restaurant_name": "Izumi Japanese Kitchen",
                "total_reviews": 15,
                "average_rating": 4.6,
                "sentiment_distribution": {
                    "positive": 13,
                    "negative": 1,
                    "neutral": 1
                },
                "aspect_analysis": {
                    "food": {
                        "mentioned_count": 14,
                        "positive_rate": 0.92,
                        "negative_rate": 0.08,
                        "neutral_rate": 0.0
                    }
                },
                "conflict_rate": 0.13,
                "strengths": ["Excellent food", "Great ambience"],
                "weaknesses": [],
                "risk_factors": [],
                "recommendation_score": 87.5
            }
        }


# ============================================
# Other Endpoint Models
# ============================================

class RestaurantListItem(BaseModel):
    """Single restaurant in list"""
    name: str = Field(..., description="Restaurant name")
    review_count: int = Field(..., description="Number of reviews")
    average_rating: float = Field(..., ge=0, le=5, description="Average rating")


class RestaurantListResponse(BaseModel):
    """Response model for /restaurants endpoint"""
    restaurants: List[RestaurantListItem] = Field(..., description="List of restaurants")
    total: int = Field(..., description="Total number of restaurants")
    
    class Config:
        json_schema_extra = {
            "example": {
                "restaurants": [
                    {
                        "name": "Izumi Japanese Kitchen",
                        "review_count": 15,
                        "average_rating": 4.6
                    }
                ],
                "total": 126
            }
        }


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str = Field(..., description="System status")
    vector_db: str = Field(..., description="Vector database status")
    total_reviews: int = Field(..., description="Total reviews loaded")
    total_restaurants: int = Field(..., description="Total restaurants")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "vector_db": "connected",
                "total_reviews": 977,
                "total_restaurants": 126,
                "models_loaded": {
                    "sentiment_model": True,
                    "embedding_model": True,
                    "intelligent_llm": True
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Restaurant not found",
                "detail": "No reviews found for the specified restaurant name"
            }
        }
