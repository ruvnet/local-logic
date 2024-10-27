import dspy
import pandas as pd
from sklearn.cluster import KMeans

class OpponentModel(dspy.Module):
    """Model opponent behavior based on historical data"""
    def __init__(self):
        super().__init__()
        self.model = KMeans(n_clusters=3)
        # For demonstration, we are not training the model with real data
    
    def analyze_opponent(self, opponent_history: str):
        # Simplified opponent tendency analysis
        if 'aggressive' in opponent_history.lower():
            return 'aggressive'
        elif 'passive' in opponent_history.lower():
            return 'passive'
        else:
            return 'neutral'
    
    def forward(self, opponent_history: str):
        tendency = self.analyze_opponent(opponent_history)
        return tendency
class OpponentModel:
    """Models opponent tendencies and adjusts strategy"""
    
    def __init__(self):
        self.style_adjustments = {
            'aggressive': {
                'raise_threshold': -0.1,  # Lower threshold against aggressive players
                'call_threshold': +0.1,   # Call more against aggressive players
                'bluff_frequency': -0.2   # Bluff less against aggressive players
            },
            'passive': {
                'raise_threshold': -0.2,  # Lower threshold against passive players
                'call_threshold': -0.1,   # Call less against passive players
                'bluff_frequency': +0.2   # Bluff more against passive players
            },
            'tight': {
                'raise_threshold': +0.1,  # Higher threshold against tight players
                'call_threshold': -0.1,   # Call less against tight players
                'bluff_frequency': +0.1   # Bluff more against tight players
            },
            'loose': {
                'raise_threshold': +0.2,  # Higher threshold against loose players
                'call_threshold': +0.2,   # Call more against loose players
                'bluff_frequency': -0.1   # Bluff less against loose players
            }
        }
    
    def analyze_opponent(self, opponent_history: str) -> dict:
        """
        Analyze opponent description and return strategy adjustments
        """
        # Default adjustments
        adjustments = {
            'raise_threshold': 0,
            'call_threshold': 0,
            'bluff_frequency': 0
        }
        
        # Apply adjustments based on opponent description keywords
        history_lower = opponent_history.lower()
        for style, style_adjusts in self.style_adjustments.items():
            if style in history_lower:
                for key in adjustments:
                    adjustments[key] += style_adjusts[key]
        
        # Add analysis of opponent's playing style
        tendency = "unknown"
        if 'aggressive' in history_lower:
            tendency = "aggressive"
        elif 'passive' in history_lower:
            tendency = "passive"
        elif 'tight' in history_lower:
            tendency = "tight"
        elif 'loose' in history_lower:
            tendency = "loose"
            
        adjustments['tendency'] = tendency
        
        return adjustments
