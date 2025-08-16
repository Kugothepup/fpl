"""
Mistral AI Team Optimizer with Blackboard Technique
Advanced FPL team optimization using Mistral AI and collaborative reasoning
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

@dataclass
class BlackboardEntry:
    """Represents an entry on the optimization blackboard"""
    agent: str
    content: Dict[str, Any]
    priority: float
    timestamp: datetime
    reasoning: str

@dataclass
class OptimizationConstraints:
    """Constraints for team optimization"""
    budget: float
    formation: str
    max_players_per_team: int = 3
    min_predicted_points: float = 0
    risk_tolerance: str = "balanced"  # conservative, balanced, aggressive

class MistralTeamOptimizer:
    """Advanced team optimizer using Mistral AI with blackboard technique"""
    
    def __init__(self):
        self.api_key = os.getenv('MISTRAL_API_KEY')
        self.api_endpoint = "https://api.mistral.ai/v1/chat/completions"
        self.blackboard: List[BlackboardEntry] = []
        self.optimization_agents = [
            "formation_specialist",
            "value_analyzer", 
            "risk_assessor",
            "fixture_analyzer",
            "form_tracker"
        ]
        
    def is_available(self) -> bool:
        """Check if Mistral API is available"""
        return bool(self.api_key)
    
    def add_to_blackboard(self, agent: str, content: Dict[str, Any], 
                         priority: float, reasoning: str):
        """Add entry to the optimization blackboard"""
        entry = BlackboardEntry(
            agent=agent,
            content=content,
            priority=priority,
            timestamp=datetime.now(),
            reasoning=reasoning
        )
        self.blackboard.append(entry)
        logger.info(f"Blackboard entry added by {agent}: {reasoning[:100]}...")
    
    def clear_blackboard(self):
        """Clear the optimization blackboard"""
        self.blackboard.clear()
        logger.info("Optimization blackboard cleared")
    
    def call_mistral(self, messages: List[Dict[str, str]], 
                    temperature: float = 0.7) -> Optional[str]:
        """Make API call to Mistral AI"""
        if not self.is_available():
            logger.warning("Mistral API key not configured")
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "mistral-large-latest",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1000
            }
            
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Mistral API call failed: {e}")
            return None
    
    def formation_analysis(self, players: List[Dict], constraints: OptimizationConstraints) -> Dict:
        """Formation specialist agent analysis"""
        formation_prompt = f"""
        As a formation specialist, analyze the optimal {constraints.formation} formation for FPL.
        
        Available players: {len(players)} total
        Budget: £{constraints.budget}M
        Formation: {constraints.formation}
        Risk tolerance: {constraints.risk_tolerance}
        
        Consider:
        1. Position balance and tactical effectiveness
        2. Budget allocation across positions
        3. Captaincy options within formation
        4. Bench strength requirements
        
        Provide a JSON response with:
        - position_priorities: ranking of position importance
        - budget_allocation: suggested spend per position
        - key_insights: tactical reasoning
        """
        
        response = self.call_mistral([
            {"role": "system", "content": "You are an expert FPL formation analyst with deep tactical knowledge."},
            {"role": "user", "content": formation_prompt}
        ])
        
        if response:
            try:
                # Extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    analysis = json.loads(response[json_start:json_end])
                    
                    self.add_to_blackboard(
                        agent="formation_specialist",
                        content=analysis,
                        priority=0.9,
                        reasoning="Formation tactical analysis completed"
                    )
                    return analysis
            except json.JSONDecodeError:
                logger.warning("Failed to parse formation analysis JSON")
        
        return {}
    
    def value_analysis(self, players: List[Dict], ml_predictions: Dict) -> Dict:
        """Value analyzer agent - finds best value players"""
        # Get top value players by position
        value_players = {}
        positions = ['GK', 'DEF', 'MID', 'FWD']
        
        for pos in positions:
            pos_players = [p for p in players if p.get('position') == pos]
            # Sort by value (predicted points / cost)
            pos_players.sort(key=lambda x: x.get('predicted_points', 0) / max(x.get('cost', 1), 0.1), reverse=True)
            value_players[pos] = pos_players[:10]  # Top 10 value players per position
        
        value_prompt = f"""
        As a value analysis expert, identify the best value players for FPL optimization.
        
        Top value players by position:
        {json.dumps({pos: [{'name': p['name'], 'cost': p['cost'], 'predicted_points': p.get('predicted_points', 0)} for p in players[:5]] for pos, players in value_players.items()}, indent=2)}
        
        Analyze:
        1. Best value picks overall
        2. Hidden gems under £6M
        3. Premium player value assessment
        4. Budget distribution strategy
        
        Provide JSON with:
        - value_picks: top recommendations per position
        - budget_strategy: how to allocate budget
        - risk_assessment: value vs safety trade-offs
        """
        
        response = self.call_mistral([
            {"role": "system", "content": "You are an expert FPL value analyst focused on points per pound optimization."},
            {"role": "user", "content": value_prompt}
        ])
        
        if response:
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    analysis = json.loads(response[json_start:json_end])
                    
                    self.add_to_blackboard(
                        agent="value_analyzer",
                        content=analysis,
                        priority=0.8,
                        reasoning="Value analysis identifies optimal price points"
                    )
                    return analysis
            except json.JSONDecodeError:
                logger.warning("Failed to parse value analysis JSON")
        
        return {}
    
    def risk_assessment(self, players: List[Dict], constraints: OptimizationConstraints) -> Dict:
        """Risk assessor agent - evaluates team risk profile"""
        risk_prompt = f"""
        As a risk assessment specialist, evaluate the risk profile for FPL team optimization.
        
        Parameters:
        - Budget: £{constraints.budget}M
        - Risk tolerance: {constraints.risk_tolerance}
        - Formation: {constraints.formation}
        
        Analyze:
        1. Injury risk players to avoid
        2. Rotation risk in squads
        3. New signing uncertainty
        4. Form vs fixtures balance
        5. Differential vs safe picks ratio
        
        Provide JSON with:
        - risk_categories: high/medium/low risk player groups
        - safe_picks: reliable foundation players
        - differentials: high-risk/high-reward options
        - overall_strategy: risk management approach
        """
        
        response = self.call_mistral([
            {"role": "system", "content": f"You are an expert FPL risk analyst with {constraints.risk_tolerance} risk tolerance."},
            {"role": "user", "content": risk_prompt}
        ])
        
        if response:
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    analysis = json.loads(response[json_start:json_end])
                    
                    self.add_to_blackboard(
                        agent="risk_assessor", 
                        content=analysis,
                        priority=0.7,
                        reasoning=f"Risk assessment for {constraints.risk_tolerance} strategy"
                    )
                    return analysis
            except json.JSONDecodeError:
                logger.warning("Failed to parse risk assessment JSON")
        
        return {}
    
    def synthesize_recommendations(self, constraints: OptimizationConstraints) -> Dict:
        """Synthesize all blackboard entries into final recommendations"""
        if not self.blackboard:
            return {"error": "No optimization data available"}
        
        # Sort blackboard by priority
        sorted_entries = sorted(self.blackboard, key=lambda x: x.priority, reverse=True)
        
        synthesis_prompt = f"""
        As the master optimizer, synthesize these agent recommendations into a final FPL team strategy:
        
        BLACKBOARD ENTRIES:
        {json.dumps([{'agent': entry.agent, 'priority': entry.priority, 'reasoning': entry.reasoning, 'content': entry.content} for entry in sorted_entries], indent=2)}
        
        CONSTRAINTS:
        - Budget: £{constraints.budget}M
        - Formation: {constraints.formation}
        - Risk tolerance: {constraints.risk_tolerance}
        
        Create a final synthesis with:
        1. Overall strategy summary
        2. Position-by-position recommendations
        3. Captain suggestions
        4. Budget allocation plan
        5. Risk mitigation strategies
        
        Provide JSON response with these sections.
        """
        
        response = self.call_mistral([
            {"role": "system", "content": "You are the master FPL optimizer who synthesizes multiple expert opinions into optimal strategy."},
            {"role": "user", "content": synthesis_prompt}
        ], temperature=0.3)  # Lower temperature for final synthesis
        
        if response:
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    synthesis = json.loads(response[json_start:json_end])
                    
                    self.add_to_blackboard(
                        agent="master_optimizer",
                        content=synthesis,
                        priority=1.0,
                        reasoning="Final strategy synthesis complete"
                    )
                    return synthesis
            except json.JSONDecodeError:
                logger.warning("Failed to parse synthesis JSON")
        
        return {"error": "Failed to synthesize recommendations"}
    
    def optimize_team_with_mistral(self, players: List[Dict], ml_predictions: Dict,
                                  constraints: OptimizationConstraints) -> Dict:
        """Main optimization method using Mistral AI with blackboard technique"""
        if not self.is_available():
            return {"error": "Mistral API not configured. Set MISTRAL_API_KEY environment variable."}
        
        logger.info("Starting Mistral AI team optimization with blackboard technique")
        self.clear_blackboard()
        
        try:
            # Run specialist agents in parallel conceptually
            formation_analysis = self.formation_analysis(players, constraints)
            value_analysis = self.value_analysis(players, ml_predictions)
            risk_analysis = self.risk_assessment(players, constraints)
            
            # Synthesize all recommendations
            final_strategy = self.synthesize_recommendations(constraints)
            
            # Return comprehensive optimization result
            return {
                "success": True,
                "strategy": final_strategy,
                "blackboard_entries": len(self.blackboard),
                "agents_consulted": list(set([entry.agent for entry in self.blackboard])),
                "optimization_method": "mistral_blackboard",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Mistral optimization failed: {e}")
            return {"error": f"Optimization failed: {str(e)}"}
    
    def get_blackboard_summary(self) -> Dict:
        """Get summary of current blackboard state"""
        return {
            "total_entries": len(self.blackboard),
            "agents": list(set([entry.agent for entry in self.blackboard])),
            "latest_entries": [
                {
                    "agent": entry.agent,
                    "reasoning": entry.reasoning,
                    "priority": entry.priority,
                    "timestamp": entry.timestamp.isoformat()
                }
                for entry in sorted(self.blackboard, key=lambda x: x.timestamp, reverse=True)[:5]
            ]
        }