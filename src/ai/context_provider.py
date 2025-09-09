#!/usr/bin/env python3
"""
Ultra AI Context Provider
Provides real-time context including time, location, and internet search for GGUF models
"""

import asyncio
import json
import requests
import datetime
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from integrations.termux_integration import termux_integration
    termux_available = True
except ImportError:
    termux_available = False

class ContextProvider:
    """Provides real-time context for AI models."""
    
    def __init__(self):
        self.cached_location = None
        self.location_cache_time = 0
        self.location_cache_duration = 300  # 5 minutes
        
        logging.info("🌍 Context Provider initialized")
    
    async def get_current_context(self) -> Dict[str, Any]:
        """Get comprehensive current context for AI models."""
        
        context = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "date": datetime.date.today().strftime("%Y-%m-%d"),
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "day_of_week": datetime.datetime.now().strftime("%A"),
            "timezone": "Local Time",
        }
        
        # Add location context
        location_info = await self.get_location_context()
        if location_info:
            context.update(location_info)
        
        return context
    
    async def get_location_context(self) -> Optional[Dict[str, Any]]:
        """Get location context including city, country, etc."""
        
        # Check cache first
        current_time = time.time()
        if (self.cached_location and 
            current_time - self.location_cache_time < self.location_cache_duration):
            return self.cached_location
        
        if not termux_available:
            return {"location_status": "GPS not available"}
        
        try:
            # Get GPS coordinates
            location_result = await termux_integration.get_location()
            
            if not location_result.get("success"):
                return {"location_status": "GPS unavailable"}
            
            location_data = location_result["data"]
            lat = location_data.get("latitude")
            lon = location_data.get("longitude")
            
            if not lat or not lon:
                return {"location_status": "No GPS fix"}
            
            # Reverse geocoding to get city/country info
            try:
                # Using a simple reverse geocoding service
                geocode_url = f"http://api.openweathermap.org/geo/1.0/reverse"
                params = {
                    "lat": lat,
                    "lon": lon,
                    "limit": 1,
                    "appid": "demo"  # You'd need a real API key for production
                }
                
                # For now, let's use a simple approach
                location_context = {
                    "latitude": lat,
                    "longitude": lon,
                    "location_status": "GPS coordinates available",
                    "coordinates": f"{lat:.4f}, {lon:.4f}"
                }
                
                # Cache the result
                self.cached_location = location_context
                self.location_cache_time = current_time
                
                return location_context
                
            except Exception as e:
                logging.warning(f"Reverse geocoding failed: {e}")
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "location_status": "GPS coordinates only"
                }
        
        except Exception as e:
            logging.error(f"Location context error: {e}")
            return {"location_status": "Location unavailable"}
    
    async def search_wikipedia(self, query: str) -> Dict[str, Any]:
        """Search Wikipedia for current information."""
        
        try:
            # Wikipedia API search
            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
            
            headers = {
                "User-Agent": "Ultra-AI/1.0 (Educational Use)"
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    "success": True,
                    "source": "Wikipedia",
                    "title": data.get("title", ""),
                    "summary": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "timestamp": datetime.datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"Wikipedia search failed with status {response.status_code}",
                    "query": query
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def search_reddit(self, query: str, subreddit: str = "all") -> Dict[str, Any]:
        """Search Reddit for current discussions."""
        
        try:
            # Reddit API search (using JSON endpoint)
            search_url = f"https://www.reddit.com/r/{subreddit}/search.json"
            
            params = {
                "q": query,
                "sort": "new",
                "limit": 5,
                "restrict_sr": "true" if subreddit != "all" else "false"
            }
            
            headers = {
                "User-Agent": "Ultra-AI/1.0 (Educational Use)"
            }
            
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                posts = []
                
                for post in data.get("data", {}).get("children", []):
                    post_data = post.get("data", {})
                    posts.append({
                        "title": post_data.get("title", ""),
                        "selftext": post_data.get("selftext", "")[:200] + "...",
                        "score": post_data.get("score", 0),
                        "subreddit": post_data.get("subreddit", ""),
                        "created_utc": post_data.get("created_utc", 0),
                        "url": f"https://reddit.com{post_data.get('permalink', '')}"
                    })
                
                return {
                    "success": True,
                    "source": "Reddit",
                    "query": query,
                    "subreddit": subreddit,
                    "posts": posts,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"Reddit search failed with status {response.status_code}",
                    "query": query
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def needs_internet_search(self, prompt: str) -> Dict[str, Any]:
        """Determine if a prompt needs internet search and what type."""
        
        prompt_lower = prompt.lower()
        
        # Current events keywords
        current_keywords = [
            "today", "now", "current", "latest", "recent", "this week", "this month",
            "2024", "2025", "what's happening", "news", "updates"
        ]
        
        # Wikipedia search keywords
        wikipedia_keywords = [
            "who is", "what is", "tell me about", "explain", "definition of",
            "history of", "biography", "facts about"
        ]
        
        # Reddit search keywords  
        reddit_keywords = [
            "discussion", "opinions", "what do people think", "reddit",
            "community", "experiences", "reviews"
        ]
        
        needs_search = any(keyword in prompt_lower for keyword in current_keywords)
        
        search_type = None
        if any(keyword in prompt_lower for keyword in wikipedia_keywords):
            search_type = "wikipedia"
        elif any(keyword in prompt_lower for keyword in reddit_keywords):
            search_type = "reddit"
        elif needs_search:
            search_type = "both"
        
        return {
            "needs_search": needs_search or search_type is not None,
            "search_type": search_type,
            "confidence": 0.8 if needs_search else 0.6
        }
    
    async def enhance_prompt_with_context(self, original_prompt: str) -> str:
        """Enhance the prompt with current context and search results if needed."""
        
        # Get current context
        context = await self.get_current_context()
        
        # Check if internet search is needed
        search_info = self.needs_internet_search(original_prompt)
        
        enhanced_prompt = f"""Current Context:
- Date: {context['date']} ({context['day_of_week']})
- Time: {context['time']}
- Location: {context.get('coordinates', 'GPS unavailable')}

"""
        
        # Add search results if needed
        if search_info["needs_search"]:
            search_results = []
            
            # Extract search query from prompt
            search_query = self._extract_search_query(original_prompt)
            
            if search_info["search_type"] in ["wikipedia", "both"]:
                wiki_result = await self.search_wikipedia(search_query)
                if wiki_result["success"]:
                    enhanced_prompt += f"""Recent Wikipedia Information about "{search_query}":
{wiki_result['summary'][:300]}...

"""
                    search_results.append("Wikipedia")
            
            if search_info["search_type"] in ["reddit", "both"]:
                reddit_result = await self.search_reddit(search_query)
                if reddit_result["success"] and reddit_result["posts"]:
                    enhanced_prompt += f"""Recent Reddit Discussions about "{search_query}":
"""
                    for post in reddit_result["posts"][:2]:  # Top 2 posts
                        enhanced_prompt += f"- {post['title']} (r/{post['subreddit']})\n"
                    enhanced_prompt += "\n"
                    search_results.append("Reddit")
            
            if search_results:
                enhanced_prompt += f"(Search results from: {', '.join(search_results)})\n\n"
        
        enhanced_prompt += f"""User Question: {original_prompt}

Please provide a comprehensive answer using the current context and any search results provided above. If you don't have current information about something, mention your knowledge cutoff and refer to the search results."""
        
        return enhanced_prompt
    
    def _extract_search_query(self, prompt: str) -> str:
        """Extract a suitable search query from the user's prompt."""
        
        # Simple extraction - remove common question words
        query = prompt.lower()
        
        # Remove common question words
        remove_words = [
            "what is", "who is", "tell me about", "explain", "how",
            "when", "where", "why", "what's", "current", "latest",
            "today", "now", "recent"
        ]
        
        for word in remove_words:
            query = query.replace(word, "").strip()
        
        # Clean up and limit length
        query = " ".join(query.split()[:4])  # Max 4 words
        
        return query if query else prompt[:50]

# Global context provider instance
context_provider = ContextProvider()

if __name__ == "__main__":
    # Test context provider
    async def test_context_provider():
        print("🌍 Ultra AI Context Provider Test")
        print("=" * 50)
        
        # Test current context
        print("📅 Getting current context...")
        context = await context_provider.get_current_context()
        print(json.dumps(context, indent=2))
        
        # Test search detection
        print("\n🔍 Testing search detection...")
        test_prompts = [
            "What's happening in the world today?",
            "Tell me about quantum computing",
            "What do people think about AI on reddit?",
            "What's the weather like?"
        ]
        
        for prompt in test_prompts:
            search_info = context_provider.needs_internet_search(prompt)
            print(f"'{prompt}' -> Needs search: {search_info['needs_search']}, Type: {search_info['search_type']}")
        
        # Test enhanced prompt
        print("\n✨ Testing prompt enhancement...")
        test_prompt = "What are the latest developments in AI?"
        enhanced = await context_provider.enhance_prompt_with_context(test_prompt)
        print(f"Original: {test_prompt}")
        print(f"Enhanced: {enhanced[:200]}...")
    
    asyncio.run(test_context_provider())