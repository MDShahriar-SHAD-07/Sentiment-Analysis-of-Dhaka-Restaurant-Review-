"""
FastAPI Restaurant Intelligence API
===================================

Main application with REST endpoints for restaurant chatbot
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any
import pandas as pd
from pathlib import Path

# Import local modules
from config import get_settings, ensure_directories
from models import (
    ChatRequest, ChatResponse, SourceDocument,
    RecommendationRequest, RecommendationResponse, RestaurantRecommendation,
    AnalysisResponse, AspectAnalysis,
    RestaurantListResponse, RestaurantListItem,
    HealthResponse, ErrorResponse
)
from data_loader import initialize_data
from intelligent_llm import IntelligentLLM, QueryIntent
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings as ChromaSettings

# Initialize settings
settings = get_settings()
ensure_directories()

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Global state
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Restaurant Intelligence API...")
    
    try:
        # Load data
        logger.info("Loading processed data...")
        df, restaurant_index = initialize_data()
        app_state['df'] = df
        app_state['restaurant_index'] = restaurant_index
        
        # Initialize embeddings
        logger.info("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        app_state['embeddings'] = embeddings
        
        # Load vector store
        logger.info("Loading vector store...")
        if settings.VECTOR_DB_PATH.exists():
            vector_store = Chroma(
                persist_directory=str(settings.VECTOR_DB_PATH),
                embedding_function=embeddings,
                client_settings=ChromaSettings(anonymized_telemetry=False)
            )
            app_state['vector_store'] = vector_store
            logger.info("✅ Vector store loaded")
        else:
            logger.warning("⚠️ Vector store not found. Some features may not work.")
            app_state['vector_store'] = None
        
        # Initialize Intelligent LLM
        logger.info("Initializing Intelligent LLM...")
        
        # Create config object for IntelligentLLM
        class Config:
            aspects = settings.ASPECTS
        
        intelligent_llm = IntelligentLLM(df, Config())
        app_state['intelligent_llm'] = intelligent_llm
        
        # Session storage
        app_state['sessions'] = {}
        
        logger.info("✅ Application startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# ============================================
# Helper Functions
# ============================================

def get_session_llm(session_id: str) -> IntelligentLLM:
    """Get or create session-specific LLM instance"""
    if session_id not in app_state['sessions']:
        # Create new session with fresh LLM instance
        df = app_state['df']
        
        class Config:
            aspects = settings.ASPECTS
        
        app_state['sessions'][session_id] = IntelligentLLM(df, Config())
    
    return app_state['sessions'][session_id]


# ============================================
# API Endpoints
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Restaurant Intelligence API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check"""
    try:
        df = app_state.get('df')
        vector_store = app_state.get('vector_store')
        
        return HealthResponse(
            status="healthy",
            vector_db="connected" if vector_store else "not_loaded",
            total_reviews=len(df) if df is not None else 0,
            total_restaurants=df['business_name'].nunique() if df is not None else 0,
            models_loaded={
                "embedding_model": app_state.get('embeddings') is not None,
                "intelligent_llm": app_state.get('intelligent_llm') is not None,
                "vector_store": vector_store is not None
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chatbot interaction endpoint
    
    Understands user queries and generates intelligent responses
    based on restaurant review data.
    """
    try:
        # Get session-specific LLM or use global one
        if request.session_id:
            llm = get_session_llm(request.session_id)
        else:
            llm = app_state['intelligent_llm']
        
        # Understand query
        understanding = llm.understand_query(request.message)
        
        # Retrieve relevant documents from vector store
        retrieved_docs = []
        if app_state.get('vector_store'):
            try:
                # Build search query
                search_query = request.message
                
                # Add restaurant filter if specific restaurant mentioned
                if understanding.restaurant_names:
                    filter_dict = {
                        "restaurant_normalized": understanding.restaurant_names[0].lower()
                    }
                    retrieved_docs = app_state['vector_store'].similarity_search(
                        search_query,
                        k=settings.RETRIEVAL_K,
                        filter=filter_dict
                    )
                else:
                    retrieved_docs = app_state['vector_store'].similarity_search(
                        search_query,
                        k=settings.RETRIEVAL_K
                    )
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # Generate response
        response_text = llm.generate_response(understanding, retrieved_docs)
        
        # Add to conversation history
        llm.add_to_history(request.message, response_text)
        
        # Format source documents
        sources = []
        for doc in retrieved_docs[:3]:  # Top 3 sources
            if hasattr(doc, 'metadata'):
                sources.append(SourceDocument(
                    content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    restaurant=doc.metadata.get('restaurant', 'Unknown'),
                    rating=doc.metadata.get('rating', 0.0),
                    sentiment=doc.metadata.get('sentiment', 'neutral')
                ))
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            metadata={
                "intent": understanding.intent.value,
                "aspects": understanding.aspects,
                "restaurant_names": understanding.restaurant_names,
                "language": understanding.language
            },
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}"
        )


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend_restaurants(request: RecommendationRequest):
    """
    Get restaurant recommendations based on filters
    
    Returns top restaurants ranked by specified criteria.
    """
    try:
        df = app_state['df']
        
        # Filter by minimum reviews
        restaurant_counts = df['business_name'].value_counts()
        valid_restaurants = restaurant_counts[restaurant_counts >= request.min_reviews].index.tolist()
        
        df_filtered = df[df['business_name'].isin(valid_restaurants)].copy()
        
        if len(df_filtered) == 0:
            return RecommendationResponse(
                recommendations=[],
                total_analyzed=0,
                filters_applied=request.dict()
            )
        
        # Calculate scores for each restaurant
        recommendations = []
        
        for restaurant in df_filtered['business_name'].unique():
            rest_df = df_filtered[df_filtered['business_name'] == restaurant]
            
            # Basic stats
            review_count = len(rest_df)
            avg_rating = rest_df['review_rating'].mean()
            sentiment_dist = rest_df['overall_sentiment'].value_counts().to_dict()
            
            # Calculate score based on aspect if specified
            if request.aspect:
                aspect_col = f"{request.aspect.value}_mentioned"
                sentiment_col = f"{request.aspect.value}_sentiment"
                
                if aspect_col in rest_df.columns:
                    aspect_df = rest_df[rest_df[aspect_col] == True]
                    
                    if len(aspect_df) == 0:
                        continue
                    
                    positive_rate = (aspect_df[sentiment_col] == request.sentiment_preference.value).mean()
                else:
                    continue
            else:
                # Overall sentiment
                positive_rate = (rest_df['overall_sentiment'] == request.sentiment_preference.value).mean()
            
            # Calculate final score (0-100)
            score = (
                positive_rate * 0.5 +
                (avg_rating / 5.0) * 0.3 +
                min(review_count / 50, 1.0) * 0.2
            ) * 100
            
            # Aspect scores
            aspect_scores = {}
            for aspect in settings.ASPECTS.keys():
                mentioned_col = f"{aspect}_mentioned"
                sentiment_col = f"{aspect}_sentiment"
                
                if mentioned_col in rest_df.columns:
                    aspect_mentioned = rest_df[mentioned_col].sum()
                    if aspect_mentioned > 0:
                        pos_rate = (rest_df[rest_df[mentioned_col]][sentiment_col] == 'positive').mean()
                        aspect_scores[aspect] = round(pos_rate * 100, 1)
            
            # Strengths and weaknesses
            strengths = [f"Excellent {k}" for k, v in aspect_scores.items() if v > 80]
            weaknesses = [f"Poor {k}" for k, v in aspect_scores.items() if v < 40]
            
            recommendations.append(RestaurantRecommendation(
                name=restaurant,
                score=round(score, 1),
                review_count=review_count,
                average_rating=round(avg_rating, 1),
                sentiment_distribution=sentiment_dist,
                aspect_scores=aspect_scores if aspect_scores else None,
                strengths=strengths,
                weaknesses=weaknesses
            ))
        
        # Sort by score
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        # Limit results
        recommendations = recommendations[:request.limit]
        
        return RecommendationResponse(
            recommendations=recommendations,
            total_analyzed=len(valid_restaurants),
            filters_applied=request.dict()
        )
        
    except Exception as e:
        logger.error(f"Recommendation endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@app.get("/analyze/{restaurant_name}", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_restaurant(restaurant_name: str):
    """
    Get detailed analysis for a specific restaurant
    
    Returns comprehensive sentiment and aspect analysis.
    """
    try:
        df = app_state['df']
        
        # Find restaurant (case-insensitive)
        restaurant_index = app_state.get('restaurant_index', {})
        normalized_name = restaurant_name.lower().strip()
        
        if normalized_name in restaurant_index:
            actual_name = restaurant_index[normalized_name]
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Restaurant '{restaurant_name}' not found"
            )
        
        rest_df = df[df['business_name'] == actual_name]
        
        if len(rest_df) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No reviews found for '{actual_name}'"
            )
        
        # Basic stats
        total_reviews = len(rest_df)
        avg_rating = rest_df['review_rating'].mean()
        sentiment_dist = rest_df['overall_sentiment'].value_counts().to_dict()
        conflict_rate = rest_df['has_conflict'].mean()
        
        # Aspect analysis
        aspect_analysis = {}
        for aspect in settings.ASPECTS.keys():
            mentioned_col = f"{aspect}_mentioned"
            sentiment_col = f"{aspect}_sentiment"
            
            if mentioned_col in rest_df.columns:
                aspect_df = rest_df[rest_df[mentioned_col] == True]
                
                if len(aspect_df) > 0:
                    total_mentioned = len(aspect_df)
                    sentiment_counts = aspect_df[sentiment_col].value_counts()
                    
                    aspect_analysis[aspect] = AspectAnalysis(
                        mentioned_count=total_mentioned,
                        positive_rate=sentiment_counts.get('positive', 0) / total_mentioned,
                        negative_rate=sentiment_counts.get('negative', 0) / total_mentioned,
                        neutral_rate=sentiment_counts.get('neutral', 0) / total_mentioned
                    )
        
        # Strengths and weaknesses
        strengths = []
        weaknesses = []
        for aspect, analysis in aspect_analysis.items():
            if analysis.positive_rate > 0.7:
                strengths.append(f"Excellent {aspect}")
            elif analysis.positive_rate < 0.4:
                weaknesses.append(f"Poor {aspect}")
        
        # Risk factors
        risk_factors = []
        if conflict_rate > 0.15:
            risk_factors.append(f"{conflict_rate*100:.0f}% reviews show mixed signals")
        
        # Calculate recommendation score
        positive_rate = sentiment_dist.get('positive', 0) / total_reviews
        recommendation_score = (
            positive_rate * 0.4 +
            (avg_rating / 5.0) * 0.3 +
            (1 - conflict_rate) * 0.2 +
            min(total_reviews / 50, 1.0) * 0.1
        ) * 100
        
        return AnalysisResponse(
            restaurant_name=actual_name,
            total_reviews=total_reviews,
            average_rating=round(avg_rating, 1),
            sentiment_distribution=sentiment_dist,
            aspect_analysis=aspect_analysis,
            conflict_rate=round(conflict_rate, 2),
            strengths=strengths,
            weaknesses=weaknesses,
            risk_factors=risk_factors,
            recommendation_score=round(recommendation_score, 1)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze restaurant: {str(e)}"
        )


@app.get("/restaurants", response_model=RestaurantListResponse, tags=["Restaurants"])
async def list_restaurants():
    """
    Get list of all restaurants with basic info
    """
    try:
        df = app_state['df']
        
        restaurants = []
        for name in sorted(df['business_name'].unique()):
            rest_df = df[df['business_name'] == name]
            
            restaurants.append(RestaurantListItem(
                name=name,
                review_count=len(rest_df),
                average_rating=round(rest_df['review_rating'].mean(), 1)
            ))
        
        return RestaurantListResponse(
            restaurants=restaurants,
            total=len(restaurants)
        )
        
    except Exception as e:
        logger.error(f"Restaurant list endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve restaurant list: {str(e)}"
        )


# ============================================
# Error Handlers
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )
