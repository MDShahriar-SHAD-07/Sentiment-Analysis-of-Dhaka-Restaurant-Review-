"""
Custom Intelligent LLM Engine for Restaurant Chatbot
=====================================================

A rule-based AI system that understands user queries and generates
intelligent, context-aware responses based on restaurant review data.

Features:
- Intent detection (recommendation, comparison, specific restaurant, general)
- Aspect extraction (food, service, price, ambience, cleanliness)
- Query understanding (restaurant names, sentiment preferences)
- Natural language response generation
- Bangla + English support
- Conversation context management
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of user intents"""
    RECOMMENDATION = "recommendation"
    COMPARISON = "comparison"
    SPECIFIC_RESTAURANT = "specific_restaurant"
    GENERAL_QUESTION = "general_question"
    GREETING = "greeting"
    UNKNOWN = "unknown"


@dataclass
class QueryUnderstanding:
    """Structured representation of understood query"""
    intent: QueryIntent
    aspects: List[str] = field(default_factory=list)
    restaurant_names: List[str] = field(default_factory=list)
    sentiment_preference: Optional[str] = None  # positive, negative, neutral
    constraints: Dict[str, Any] = field(default_factory=dict)
    original_query: str = ""
    language: str = "en"  # en or bn


class IntelligentLLM:
    """
    Custom Intelligent LLM Engine
    
    This is a rule-based AI system that mimics LLM behavior by:
    1. Understanding user intent through pattern matching
    2. Extracting relevant information (aspects, restaurants, preferences)
    3. Generating natural language responses using templates + dynamic data
    """
    
    def __init__(self, df: pd.DataFrame, config: Any):
        self.df = df
        self.config = config
        self.conversation_history: List[Dict[str, str]] = []
        
        # Build restaurant index for fast lookup
        self.restaurant_index = self._build_restaurant_index()
        
        # Intent patterns
        self.intent_patterns = self._initialize_intent_patterns()
        
        # Aspect keywords (English + Bangla)
        self.aspect_keywords = {
            "food": ["food", "meal", "dish", "cuisine", "taste", "flavor", "delicious", 
                     "খাবার", "স্বাদ", "খাদ্য", "রান্না"],
            "service": ["service", "staff", "waiter", "waitress", "server", "manager",
                       "সার্ভিস", "কর্মী", "সেবা"],
            "price": ["price", "cost", "expensive", "cheap", "affordable", "value", "budget",
                     "দাম", "মূল্য", "খরচ"],
            "ambience": ["ambience", "atmosphere", "environment", "decor", "vibe", "setting",
                        "পরিবেশ", "আবহাওয়া"],
            "cleanliness": ["clean", "hygiene", "sanitary", "dirty", "neat",
                           "পরিষ্কার", "স্বাস্থ্যকর"]
        }
        
        # Sentiment keywords
        self.sentiment_keywords = {
            "positive": ["best", "good", "great", "excellent", "amazing", "top", "recommend",
                        "ভালো", "সেরা", "উত্তম"],
            "negative": ["worst", "bad", "poor", "terrible", "avoid", "not good",
                        "খারাপ", "বাজে"]
        }
        
    def _build_restaurant_index(self) -> Dict[str, str]:
        """Build normalized restaurant name index"""
        index = {}
        for name in self.df['business_name'].unique():
            normalized = name.lower().strip()
            index[normalized] = name
            # Add partial matches
            words = normalized.split()
            for word in words:
                if len(word) > 3:  # Avoid short words
                    index[word] = name
        return index
    
    def _initialize_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Initialize regex patterns for intent detection"""
        return {
            QueryIntent.RECOMMENDATION: [
                r"(which|what|suggest|recommend|best|top|good)\s+(restaurant|place)",
                r"(where|which place)\s+(can|should|to)\s+(i|we)",
                r"(কোন|কোনটি|সেরা|ভালো)\s+(রেস্টুরেন্ট|জায়গা)",
                r"recommend.*restaurant",
                r"best.*for",
            ],
            QueryIntent.COMPARISON: [
                r"(compare|difference|vs|versus|better)",
                r"(which is better|which one)",
                r"(তুলনা|পার্থক্য)",
            ],
            QueryIntent.SPECIFIC_RESTAURANT: [
                r"(tell me about|what about|how is|review of)",
                r"(সম্পর্কে বলো|কেমন)",
            ],
            QueryIntent.GREETING: [
                r"^(hi|hello|hey|greetings|good morning|good evening)",
                r"^(হাই|হ্যালো|নমস্কার|আসসালামু আলাইকুম)",
            ],
        }
    
    def understand_query(self, query: str) -> QueryUnderstanding:
        """
        Understand user query and extract structured information
        
        Args:
            query: User's natural language query
            
        Returns:
            QueryUnderstanding object with extracted information
        """
        query_lower = query.lower().strip()
        
        # Detect language
        language = "bn" if self._contains_bangla(query) else "en"
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Extract aspects
        aspects = self._extract_aspects(query_lower)
        
        # Extract restaurant names
        restaurant_names = self._extract_restaurant_names(query_lower)
        
        # Detect sentiment preference
        sentiment_preference = self._detect_sentiment_preference(query_lower)
        
        # Extract constraints
        constraints = self._extract_constraints(query_lower)
        
        return QueryUnderstanding(
            intent=intent,
            aspects=aspects,
            restaurant_names=restaurant_names,
            sentiment_preference=sentiment_preference,
            constraints=constraints,
            original_query=query,
            language=language
        )
    
    def _contains_bangla(self, text: str) -> bool:
        """Check if text contains Bangla characters"""
        return bool(re.search(r'[\u0980-\u09FF]', text))
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect user intent from query"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        
        # Default to general question
        return QueryIntent.GENERAL_QUESTION
    
    def _extract_aspects(self, query: str) -> List[str]:
        """Extract mentioned aspects from query"""
        mentioned_aspects = []
        for aspect, keywords in self.aspect_keywords.items():
            if any(keyword in query for keyword in keywords):
                mentioned_aspects.append(aspect)
        return mentioned_aspects
    
    def _extract_restaurant_names(self, query: str) -> List[str]:
        """Extract restaurant names from query"""
        found_restaurants = []
        for normalized_name, original_name in self.restaurant_index.items():
            if normalized_name in query:
                if original_name not in found_restaurants:
                    found_restaurants.append(original_name)
        return found_restaurants
    
    def _detect_sentiment_preference(self, query: str) -> Optional[str]:
        """Detect if user wants positive or negative reviews"""
        for sentiment, keywords in self.sentiment_keywords.items():
            if any(keyword in query for keyword in keywords):
                return sentiment
        return None
    
    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract constraints like 'cheap', 'couples', 'family' etc."""
        constraints = {}
        
        # Price constraints
        if any(word in query for word in ["cheap", "affordable", "budget", "সস্তা"]):
            constraints["price_preference"] = "low"
        elif any(word in query for word in ["expensive", "premium", "luxury", "দামি"]):
            constraints["price_preference"] = "high"
        
        # Occasion constraints
        if any(word in query for word in ["couple", "romantic", "date", "রোমান্টিক"]):
            constraints["occasion"] = "romantic"
        elif any(word in query for word in ["family", "kids", "children", "পরিবার"]):
            constraints["occasion"] = "family"
        elif any(word in query for word in ["business", "meeting", "corporate"]):
            constraints["occasion"] = "business"
        
        return constraints
    
    def generate_response(self, understanding: QueryUnderstanding, 
                         retrieved_docs: List[Any] = None) -> str:
        """
        Generate natural language response based on query understanding
        
        Args:
            understanding: QueryUnderstanding object
            retrieved_docs: Optional retrieved documents from vector store
            
        Returns:
            Natural language response string
        """
        intent = understanding.intent
        
        if intent == QueryIntent.GREETING:
            return self._generate_greeting_response(understanding)
        
        elif intent == QueryIntent.RECOMMENDATION:
            return self._generate_recommendation_response(understanding, retrieved_docs)
        
        elif intent == QueryIntent.SPECIFIC_RESTAURANT:
            return self._generate_specific_restaurant_response(understanding)
        
        elif intent == QueryIntent.COMPARISON:
            return self._generate_comparison_response(understanding)
        
        else:
            return self._generate_general_response(understanding, retrieved_docs)
    
    def _generate_greeting_response(self, understanding: QueryUnderstanding) -> str:
        """Generate greeting response"""
        if understanding.language == "bn":
            return ("নমস্কার! আমি ঢাকার রেস্টুরেন্ট সম্পর্কে আপনাকে সাহায্য করতে পারি। "
                   "আপনি কোন রেস্টুরেন্ট সম্পর্কে জানতে চান?")
        else:
            return ("Hello! I'm your restaurant intelligence assistant for Dhaka restaurants. "
                   "I can help you find the best places to eat based on real customer reviews. "
                   "What would you like to know?")
    
    def _generate_recommendation_response(self, understanding: QueryUnderstanding,
                                         retrieved_docs: List[Any]) -> str:
        """Generate recommendation response with statistics"""
        aspects = understanding.aspects
        sentiment_pref = understanding.sentiment_preference or "positive"
        
        # If specific aspect mentioned, filter by that
        if aspects:
            primary_aspect = aspects[0]
            return self._get_top_restaurants_by_aspect(primary_aspect, sentiment_pref,
                                                       understanding.constraints)
        else:
            # General recommendation
            return self._get_top_restaurants_overall(sentiment_pref, understanding.constraints)
    
    def _get_top_restaurants_by_aspect(self, aspect: str, sentiment: str,
                                       constraints: Dict) -> str:
        """Get top restaurants for specific aspect"""
        aspect_col = f"{aspect}_mentioned"
        sentiment_col = f"{aspect}_sentiment"
        
        # Filter restaurants with this aspect mentioned
        df_filtered = self.df[self.df[aspect_col] == True].copy()
        
        if len(df_filtered) == 0:
            return f"I don't have enough data about {aspect} to make a recommendation."
        
        # Calculate scores by restaurant
        restaurant_scores = []
        for restaurant in df_filtered['business_name'].unique():
            rest_df = df_filtered[df_filtered['business_name'] == restaurant]
            
            if len(rest_df) < 3:  # Minimum reviews threshold
                continue
            
            # Calculate positive sentiment rate
            positive_rate = (rest_df[sentiment_col] == sentiment).mean()
            review_count = len(rest_df)
            
            # Confidence score (more reviews = higher confidence)
            confidence = min(review_count / 20, 1.0)
            
            # Final score
            score = positive_rate * 0.7 + confidence * 0.3
            
            restaurant_scores.append({
                'name': restaurant,
                'score': score,
                'positive_rate': positive_rate,
                'review_count': review_count
            })
        
        # Sort by score
        restaurant_scores.sort(key=lambda x: x['score'], reverse=True)
        
        if not restaurant_scores:
            return f"Insufficient data to recommend restaurants based on {aspect}."
        
        # Generate response
        top_restaurant = restaurant_scores[0]
        
        response = (f"Based on {len(df_filtered)} {aspect}-related reviews, "
                   f"**{top_restaurant['name']}** has the highest {sentiment} sentiment "
                   f"({top_restaurant['positive_rate']*100:.0f}%) with "
                   f"{top_restaurant['review_count']} reviews mentioning {aspect}.")
        
        # Add top 3 if available
        if len(restaurant_scores) >= 3:
            response += "\n\n**Top 3 Recommendations:**\n"
            for i, rest in enumerate(restaurant_scores[:3], 1):
                response += (f"{i}. {rest['name']} - "
                           f"{rest['positive_rate']*100:.0f}% positive "
                           f"({rest['review_count']} reviews)\n")
        
        return response
    
    def _get_top_restaurants_overall(self, sentiment: str, constraints: Dict) -> str:
        """Get top restaurants based on overall sentiment"""
        # Calculate overall scores
        restaurant_scores = []
        
        for restaurant in self.df['business_name'].unique():
            rest_df = self.df[self.df['business_name'] == restaurant]
            
            if len(rest_df) < 3:
                continue
            
            # Overall sentiment rate
            sentiment_rate = (rest_df['overall_sentiment'] == sentiment).mean()
            review_count = len(rest_df)
            avg_rating = rest_df['review_rating'].mean()
            
            # Conflict rate (lower is better)
            conflict_rate = rest_df['has_conflict'].mean()
            
            # Score calculation
            score = (sentiment_rate * 0.4 + 
                    (avg_rating / 5.0) * 0.3 + 
                    (1 - conflict_rate) * 0.2 +
                    min(review_count / 50, 1.0) * 0.1)
            
            restaurant_scores.append({
                'name': restaurant,
                'score': score,
                'sentiment_rate': sentiment_rate,
                'review_count': review_count,
                'avg_rating': avg_rating
            })
        
        restaurant_scores.sort(key=lambda x: x['score'], reverse=True)
        
        if not restaurant_scores:
            return "Insufficient data to make recommendations."
        
        top = restaurant_scores[0]
        
        response = (f"Based on analysis of {len(self.df)} reviews across "
                   f"{self.df['business_name'].nunique()} restaurants:\n\n"
                   f"**Top Recommendation: {top['name']}**\n"
                   f"- Overall {sentiment} sentiment: {top['sentiment_rate']*100:.0f}%\n"
                   f"- Average rating: {top['avg_rating']:.1f}/5.0\n"
                   f"- Based on {top['review_count']} reviews")
        
        # Add top 5
        if len(restaurant_scores) >= 5:
            response += "\n\n**Top 5 Overall:**\n"
            for i, rest in enumerate(restaurant_scores[:5], 1):
                response += (f"{i}. {rest['name']} - {rest['avg_rating']:.1f}⭐ "
                           f"({rest['review_count']} reviews)\n")
        
        return response
    
    def _generate_specific_restaurant_response(self, understanding: QueryUnderstanding) -> str:
        """Generate detailed analysis for specific restaurant"""
        if not understanding.restaurant_names:
            return "Please specify which restaurant you'd like to know about."
        
        restaurant_name = understanding.restaurant_names[0]
        rest_df = self.df[self.df['business_name'] == restaurant_name]
        
        if len(rest_df) == 0:
            return f"I don't have any reviews for '{restaurant_name}'. Please check the spelling."
        
        # Calculate statistics
        total_reviews = len(rest_df)
        avg_rating = rest_df['review_rating'].mean()
        sentiment_dist = rest_df['overall_sentiment'].value_counts()
        
        # Aspect analysis
        aspect_summary = []
        for aspect in self.config.aspects.keys():
            mentioned_col = f"{aspect}_mentioned"
            sentiment_col = f"{aspect}_sentiment"
            
            if mentioned_col in rest_df.columns:
                mentioned = rest_df[mentioned_col].sum()
                if mentioned > 0:
                    positive_rate = (rest_df[rest_df[mentioned_col]][sentiment_col] == 'positive').mean()
                    aspect_summary.append(f"  - {aspect.title()}: {positive_rate*100:.0f}% positive ({mentioned} mentions)")
        
        # Conflict analysis
        conflict_rate = rest_df['has_conflict'].mean()
        
        # Generate response
        response = f"**{restaurant_name}** - Detailed Analysis\n\n"
        response += f"📊 **Overall Statistics:**\n"
        response += f"- Total Reviews: {total_reviews}\n"
        response += f"- Average Rating: {avg_rating:.1f}/5.0\n"
        response += f"- Sentiment: {sentiment_dist.get('positive', 0)} positive, "
        response += f"{sentiment_dist.get('negative', 0)} negative, "
        response += f"{sentiment_dist.get('neutral', 0)} neutral\n\n"
        
        if aspect_summary:
            response += "🎯 **Aspect Breakdown:**\n"
            response += "\n".join(aspect_summary) + "\n\n"
        
        if conflict_rate > 0.15:
            response += f"⚠️ **Note:** {conflict_rate*100:.0f}% of reviews show mixed signals (e.g., high rating but negative comments)\n"
        
        return response
    
    def _generate_comparison_response(self, understanding: QueryUnderstanding) -> str:
        """Generate comparison between restaurants"""
        if len(understanding.restaurant_names) < 2:
            return "Please specify at least 2 restaurants to compare."
        
        comparison_data = []
        
        for restaurant in understanding.restaurant_names[:3]:  # Max 3 restaurants
            rest_df = self.df[self.df['business_name'] == restaurant]
            
            if len(rest_df) == 0:
                continue
            
            comparison_data.append({
                'name': restaurant,
                'reviews': len(rest_df),
                'avg_rating': rest_df['review_rating'].mean(),
                'positive_rate': (rest_df['overall_sentiment'] == 'positive').mean(),
                'conflict_rate': rest_df['has_conflict'].mean()
            })
        
        if len(comparison_data) < 2:
            return "Insufficient data to compare these restaurants."
        
        # Generate comparison
        response = "**Restaurant Comparison:**\n\n"
        
        for data in comparison_data:
            response += f"**{data['name']}**\n"
            response += f"- Reviews: {data['reviews']}\n"
            response += f"- Avg Rating: {data['avg_rating']:.1f}/5.0\n"
            response += f"- Positive Sentiment: {data['positive_rate']*100:.0f}%\n"
            response += f"- Reliability: {(1-data['conflict_rate'])*100:.0f}%\n\n"
        
        # Winner
        winner = max(comparison_data, key=lambda x: x['avg_rating'])
        response += f"🏆 **Best Overall:** {winner['name']} ({winner['avg_rating']:.1f}⭐)"
        
        return response
    
    def _generate_general_response(self, understanding: QueryUnderstanding,
                                   retrieved_docs: List[Any]) -> str:
        """Generate general response based on retrieved documents"""
        if not retrieved_docs or len(retrieved_docs) == 0:
            return ("I don't have enough specific information to answer that question. "
                   "Try asking about specific restaurants or aspects like food, service, or price.")
        
        # Analyze retrieved documents
        doc_count = len(retrieved_docs)
        
        response = f"Based on {doc_count} relevant reviews:\n\n"
        
        # Extract key insights from metadata
        restaurants = set()
        sentiments = []
        
        for doc in retrieved_docs[:5]:  # Top 5
            if hasattr(doc, 'metadata'):
                restaurants.add(doc.metadata.get('restaurant', 'Unknown'))
                sentiments.append(doc.metadata.get('sentiment', 'neutral'))
        
        if restaurants:
            response += f"**Mentioned Restaurants:** {', '.join(list(restaurants)[:3])}\n\n"
        
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        
        if positive_count > negative_count:
            response += f"Overall sentiment is **positive** ({positive_count}/{len(sentiments)} reviews)"
        elif negative_count > positive_count:
            response += f"Overall sentiment is **negative** ({negative_count}/{len(sentiments)} reviews)"
        else:
            response += "Sentiment is **mixed** across reviews"
        
        return response
    
    def add_to_history(self, user_message: str, bot_response: str):
        """Add conversation to history"""
        self.conversation_history.append({
            "user": user_message,
            "bot": bot_response
        })
        
        # Keep only last 10 exchanges
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_context(self) -> str:
        """Get conversation context for multi-turn dialogues"""
        if not self.conversation_history:
            return ""
        
        context = "Previous conversation:\n"
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            context += f"User: {exchange['user']}\n"
            context += f"Bot: {exchange['bot'][:100]}...\n"
        
        return context
