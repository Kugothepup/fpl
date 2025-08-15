#!/usr/bin/env python3
"""
FPL Manager v3 - News Service
Real news integration via Perplexity AI for player insights
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import Config

logger = logging.getLogger(__name__)

@dataclass
class NewsImpact:
    """Analysis of how news affects FPL decisions"""
    player_id: int
    player_name: str
    impact_type: str  # 'injury', 'suspension', 'form', 'international', 'transfer'
    impact_severity: str  # 'low', 'medium', 'high', 'critical'
    impact_duration: str  # 'short_term', 'medium_term', 'long_term', 'season_ending'
    fpl_recommendation: str  # 'avoid', 'monitor', 'consider', 'target'
    confidence: float  # 0.0 to 1.0
    reasoning: str

@dataclass
class NewsArticle:
    """News article with FPL relevance"""
    title: str
    summary: str
    source: str
    timestamp: str
    relevance_score: float
    players_mentioned: List[str]
    teams_mentioned: List[str]
    category: str  # 'injury', 'transfer', 'form', 'tactical', 'general'
    fpl_impact: Optional[NewsImpact] = None

class NewsService:
    """Service for fetching and analyzing Premier League news via Perplexity AI"""
    
    def __init__(self):
        self.api_key = Config.PERPLEXITY_API_KEY
        self.api_url = Config.PERPLEXITY_API_URL
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}' if self.api_key else '',
            'Content-Type': 'application/json',
            'User-Agent': 'FPL-Manager-v3-News/1.0'
        })
        
        # Keywords for different types of news
        self.news_keywords = {
            'injury': ['injury', 'injured', 'fitness', 'strain', 'tear', 'surgery', 'medical', 'physio'],
            'suspension': ['suspended', 'ban', 'red card', 'disciplinary', 'appeal'],
            'international': ['international', 'national team', 'world cup', 'euros', 'qualifier'],
            'transfer': ['transfer', 'signing', 'sold', 'bought', 'contract', 'loan'],
            'form': ['goal', 'assist', 'performance', 'man of the match', 'player of the month'],
            'tactical': ['formation', 'position', 'role', 'starting eleven', 'rotation']
        }
        
        # Premier League teams for context
        self.premier_league_teams = list(Config.PREMIER_LEAGUE_TEAMS.values())
        
        logger.info(f"News Service initialized. Perplexity API available: {bool(self.api_key)}")
    
    def get_latest_fpl_news(self, hours_back: int = 24, max_articles: int = 10) -> List[NewsArticle]:
        """Get latest Premier League news relevant to FPL"""
        if not self.api_key:
            logger.warning("Perplexity API key not configured")
            return self._get_fallback_news()
        
        try:
            # Create targeted query for FPL-relevant news
            query = self._build_news_query(hours_back)
            
            response = self._query_perplexity(query)
            
            if not response:
                return self._get_fallback_news()
            
            # Parse and analyze the response
            articles = self._parse_news_response(response)
            
            # Limit results
            articles = articles[:max_articles]
            
            logger.info(f"Retrieved {len(articles)} FPL-relevant news articles")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch FPL news: {e}")
            return self._get_fallback_news()
    
    def analyze_player_news(self, player_name: str, team_name: str = None) -> List[NewsImpact]:
        """Get news analysis for a specific player"""
        if not self.api_key:
            return []
        
        try:
            # Create player-specific query
            query = f"""
            Recent news about {player_name} {f'from {team_name}' if team_name else ''} in the Premier League:
            - Injury status and fitness updates
            - Recent performance and form
            - Transfer rumors or contract news
            - International duty commitments
            - Any disciplinary issues or suspensions
            
            Please provide specific, factual information with dates where possible.
            Focus on information that would impact Fantasy Premier League decisions.
            """
            
            response = self._query_perplexity(query)
            
            if not response:
                return []
            
            # Analyze the response for FPL impact
            impacts = self._analyze_player_impact(response, player_name)
            
            logger.info(f"Analyzed news impact for {player_name}: {len(impacts)} findings")
            return impacts
            
        except Exception as e:
            logger.error(f"Failed to analyze player news for {player_name}: {e}")
            return []
    
    def get_injury_updates(self) -> List[NewsImpact]:
        """Get comprehensive injury updates for Premier League players"""
        if not self.api_key:
            return []
        
        try:
            query = """
            Current injury list and fitness updates for Premier League players:
            - Players currently injured with expected return dates
            - Players returning from injury soon
            - Fitness concerns or doubts for upcoming matches
            - Surgery or long-term injury updates
            
            Please provide player names, teams, injury types, and expected return timeframes.
            Focus on information relevant for Fantasy Premier League team selection.
            """
            
            response = self._query_perplexity(query)
            
            if not response:
                return []
            
            # Parse injury information
            injury_impacts = self._parse_injury_updates(response)
            
            logger.info(f"Retrieved {len(injury_impacts)} injury updates")
            return injury_impacts
            
        except Exception as e:
            logger.error(f"Failed to get injury updates: {e}")
            return []
    
    def get_team_news(self, team_name: str) -> List[NewsArticle]:
        """Get news for a specific team"""
        if not self.api_key:
            return []
        
        try:
            query = f"""
            Recent news about {team_name} in the Premier League:
            - Team selection and lineup changes
            - Tactical changes or formation updates
            - Player injuries and fitness updates
            - Transfer activity
            - Manager quotes about player availability
            
            Focus on information that affects Fantasy Premier League decisions.
            """
            
            response = self._query_perplexity(query)
            
            if not response:
                return []
            
            articles = self._parse_team_news(response, team_name)
            
            logger.info(f"Retrieved {len(articles)} news articles for {team_name}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to get team news for {team_name}: {e}")
            return []
    
    def _build_news_query(self, hours_back: int) -> str:
        """Build optimized query for FPL news"""
        time_filter = f"in the last {hours_back} hours" if hours_back <= 48 else f"in the last {hours_back//24} days"
        
        query = f"""
        Latest Premier League news {time_filter} that affects Fantasy Premier League decisions:
        
        1. Player injuries, fitness updates, and return dates
        2. Suspensions, disciplinary actions, and bans
        3. Transfer news and new signings
        4. Team selection updates and rotation news
        5. International duty and player availability
        6. Manager quotes about player fitness and selection
        
        Please provide:
        - Player names and teams
        - Specific details about the situation
        - Expected impact duration
        - Source credibility
        
        Focus only on factual, confirmed information from reliable sources.
        Exclude speculation and rumors unless clearly marked as such.
        """
        
        return query
    
    def _query_perplexity(self, query: str) -> Optional[str]:
        """Send query to Perplexity AI and get response"""
        try:
            payload = {
                'model': 'llama-3.1-sonar-small-128k-online',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a Premier League news analyst specializing in Fantasy Premier League insights. Provide factual, up-to-date information with specific details and dates.'
                    },
                    {
                        'role': 'user',
                        'content': query
                    }
                ],
                'temperature': 0.2,
                'max_tokens': 2000
            }
            
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            
            logger.warning("Invalid Perplexity response format")
            return None
            
        except requests.RequestException as e:
            logger.error(f"Perplexity API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error querying Perplexity: {e}")
            return None
    
    def _parse_news_response(self, response: str) -> List[NewsArticle]:
        """Parse Perplexity response into NewsArticle objects"""
        try:
            articles = []
            
            # Split response into sections
            sections = response.split('\n\n')
            
            for section in sections:
                if len(section.strip()) < 50:  # Skip short sections
                    continue
                
                # Extract key information
                article = self._extract_article_info(section)
                if article:
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to parse news response: {e}")
            return []
    
    def _extract_article_info(self, text: str) -> Optional[NewsArticle]:
        """Extract article information from text"""
        try:
            # Simple parsing - could be enhanced with NLP
            lines = text.strip().split('\n')
            
            # Try to identify title (usually first line or line with certain patterns)
            title = lines[0] if lines else "Premier League Update"
            
            # Categorize the news
            category = self._categorize_news(text)
            
            # Extract mentioned players and teams
            players_mentioned = self._extract_players(text)
            teams_mentioned = self._extract_teams(text)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance(text, players_mentioned, teams_mentioned)
            
            # Create article
            article = NewsArticle(
                title=title[:100],  # Limit title length
                summary=text[:500],  # Limit summary length
                source="Perplexity AI Research",
                timestamp=datetime.now().isoformat(),
                relevance_score=relevance_score,
                players_mentioned=players_mentioned,
                teams_mentioned=teams_mentioned,
                category=category
            )
            
            # Add FPL impact analysis if high relevance
            if relevance_score > 0.7 and players_mentioned:
                article.fpl_impact = self._analyze_fpl_impact(text, players_mentioned[0])
            
            return article
            
        except Exception as e:
            logger.error(f"Failed to extract article info: {e}")
            return None
    
    def _categorize_news(self, text: str) -> str:
        """Categorize news based on content"""
        text_lower = text.lower()
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in self.news_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or 'general' if none
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    def _extract_players(self, text: str) -> List[str]:
        """Extract player names from text (simplified)"""
        # This is a basic implementation - could be enhanced with NER
        players = []
        
        # Look for common name patterns
        # This would ideally use a database of current Premier League players
        name_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
            r'\b[A-Z]\. [A-Z][a-z]+\b',      # F. Last
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            players.extend(matches)
        
        # Filter out common false positives
        false_positives = ['Premier League', 'Fantasy Premier', 'Manager', 'Team', 'Player']
        players = [p for p in players if p not in false_positives]
        
        return list(set(players))[:5]  # Limit and deduplicate
    
    def _extract_teams(self, text: str) -> List[str]:
        """Extract team names from text"""
        teams = []
        
        for team in self.premier_league_teams:
            if team.lower() in text.lower():
                teams.append(team)
        
        return list(set(teams))
    
    def _calculate_relevance(self, text: str, players: List[str], teams: List[str]) -> float:
        """Calculate FPL relevance score for news"""
        score = 0.0
        text_lower = text.lower()
        
        # Base score for FPL keywords
        fpl_keywords = ['fantasy', 'fpl', 'points', 'captain', 'transfer']
        score += sum(0.1 for keyword in fpl_keywords if keyword in text_lower)
        
        # Score for player mentions
        score += len(players) * 0.15
        
        # Score for team mentions
        score += len(teams) * 0.1
        
        # Score for impact keywords
        impact_keywords = ['injury', 'suspended', 'return', 'fit', 'doubt', 'ruled out']
        score += sum(0.2 for keyword in impact_keywords if keyword in text_lower)
        
        # Bonus for specific impact indicators
        if any(keyword in text_lower for keyword in ['will miss', 'out for', 'expected return']):
            score += 0.3
        
        return min(1.0, score)
    
    def _analyze_fpl_impact(self, text: str, player_name: str) -> Optional[NewsImpact]:
        """Analyze FPL impact of news for a specific player"""
        try:
            text_lower = text.lower()
            
            # Determine impact type
            impact_type = 'general'
            if any(keyword in text_lower for keyword in self.news_keywords['injury']):
                impact_type = 'injury'
            elif any(keyword in text_lower for keyword in self.news_keywords['suspension']):
                impact_type = 'suspension'
            elif any(keyword in text_lower for keyword in self.news_keywords['international']):
                impact_type = 'international'
            elif any(keyword in text_lower for keyword in self.news_keywords['transfer']):
                impact_type = 'transfer'
            elif any(keyword in text_lower for keyword in self.news_keywords['form']):
                impact_type = 'form'
            
            # Determine severity
            severity = 'low'
            if any(phrase in text_lower for phrase in ['ruled out', 'surgery', 'long term', 'season ending']):
                severity = 'critical'
            elif any(phrase in text_lower for phrase in ['will miss', 'doubt', 'concern']):
                severity = 'high'
            elif any(phrase in text_lower for phrase in ['minor', 'precaution', 'assessment']):
                severity = 'medium'
            
            # Determine duration
            duration = 'short_term'
            if any(phrase in text_lower for phrase in ['season', 'months']):
                duration = 'season_ending' if 'season' in text_lower else 'long_term'
            elif any(phrase in text_lower for phrase in ['weeks', 'month']):
                duration = 'medium_term'
            
            # Generate recommendation
            if severity == 'critical':
                recommendation = 'avoid'
            elif severity == 'high':
                recommendation = 'monitor'
            elif impact_type == 'form' and 'good' in text_lower:
                recommendation = 'consider'
            else:
                recommendation = 'monitor'
            
            # Calculate confidence
            confidence = 0.6  # Base confidence
            if any(word in text_lower for word in ['confirmed', 'official', 'announced']):
                confidence += 0.2
            if any(word in text_lower for word in ['rumor', 'speculation', 'might']):
                confidence -= 0.3
            
            confidence = max(0.1, min(1.0, confidence))
            
            return NewsImpact(
                player_id=0,  # Would need player database lookup
                player_name=player_name,
                impact_type=impact_type,
                impact_severity=severity,
                impact_duration=duration,
                fpl_recommendation=recommendation,
                confidence=confidence,
                reasoning=text[:200] + "..." if len(text) > 200 else text
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze FPL impact: {e}")
            return None
    
    def _analyze_player_impact(self, response: str, player_name: str) -> List[NewsImpact]:
        """Analyze multiple impacts for a player"""
        impacts = []
        
        # Split response into relevant sections
        sections = response.split('\n\n')
        
        for section in sections:
            if player_name.lower() in section.lower():
                impact = self._analyze_fpl_impact(section, player_name)
                if impact:
                    impacts.append(impact)
        
        return impacts
    
    def _parse_injury_updates(self, response: str) -> List[NewsImpact]:
        """Parse injury-specific updates"""
        impacts = []
        
        lines = response.split('\n')
        current_player = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Look for player mentions
            players = self._extract_players(line)
            if players:
                current_player = players[0]
            
            # If we have injury-related content and a current player
            if current_player and any(keyword in line.lower() for keyword in self.news_keywords['injury']):
                impact = self._analyze_fpl_impact(line, current_player)
                if impact:
                    impacts.append(impact)
        
        return impacts
    
    def _parse_team_news(self, response: str, team_name: str) -> List[NewsArticle]:
        """Parse team-specific news"""
        articles = []
        
        sections = response.split('\n\n')
        
        for section in sections:
            if team_name.lower() in section.lower():
                article = self._extract_article_info(section)
                if article:
                    articles.append(article)
        
        return articles
    
    def _get_fallback_news(self) -> List[NewsArticle]:
        """Return fallback news when API is unavailable"""
        return [
            NewsArticle(
                title="Premier League News Integration",
                summary="Add PERPLEXITY_API_KEY to your .env file to enable real-time Premier League news analysis for injury updates, transfer news, and team selection insights.",
                source="FPL Manager System",
                timestamp=datetime.now().isoformat(),
                relevance_score=0.5,
                players_mentioned=[],
                teams_mentioned=[],
                category="system"
            ),
            NewsArticle(
                title="News Service Configuration",
                summary="The news service uses Perplexity AI to fetch and analyze Premier League news that impacts FPL decisions. Configure your API key to access real-time updates.",
                source="FPL Manager System",
                timestamp=datetime.now().isoformat(),
                relevance_score=0.5,
                players_mentioned=[],
                teams_mentioned=[],
                category="system"
            )
        ]