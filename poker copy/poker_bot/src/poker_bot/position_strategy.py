import dspy

class PositionStrategy(dspy.Module):
    """Determine optimal strategy based on position and stack sizes"""
    def __init__(self):
        super().__init__()
        self.strategy = dspy.Function(self.determine_strategy)
    
    def determine_strategy(self, position: str, hand_strength: float, stack_size: float, opponent_stack: float):
        # Simplified strategy logic
        if position == 'BTN' and hand_strength > 0.5:
            return 'raise'
        elif position == 'SB' and hand_strength > 0.7:
            return 'raise'
        else:
            return 'call' if hand_strength > 0.3 else 'fold'
    
    def forward(self, position: str, hand_strength: float, stack_size: float, opponent_stack: float):
        action = self.strategy(
            position=position,
            hand_strength=hand_strength,
            stack_size=stack_size,
            opponent_stack=opponent_stack
        )
        return action
class PositionStrategy:
    """Position-based strategy adjustments"""
    
    def __init__(self):
        self.position_ranges = {
            'BTN': {'raise_range': 0.3, 'call_range': 0.4},  # Button - widest ranges
            'CO': {'raise_range': 0.35, 'call_range': 0.45},  # Cutoff
            'MP': {'raise_range': 0.4, 'call_range': 0.5},   # Middle position
            'UTG': {'raise_range': 0.5, 'call_range': 0.6},  # Under the gun - tightest ranges
            'SB': {'raise_range': 0.45, 'call_range': 0.55}, # Small blind
            'BB': {'raise_range': 0.4, 'call_range': 0.5}    # Big blind
        }
    
    def get_position_adjustment(self, position: str, hand_strength: float) -> dict:
        """
        Get position-based strategy adjustments
        Returns dict with recommended actions and ranges
        """
        if position not in self.position_ranges:
            position = 'MP'  # Default to middle position if unknown
            
        ranges = self.position_ranges[position]
        
        # Determine recommended action based on hand strength and position
        if hand_strength >= ranges['raise_range']:
            action = 'raise'
            reason = f"Strong hand ({hand_strength:.2%}) in {position}"
        elif hand_strength >= ranges['call_range']:
            action = 'call'
            reason = f"Playable hand ({hand_strength:.2%}) in {position}"
        else:
            action = 'fold'
            reason = f"Hand too weak ({hand_strength:.2%}) for {position}"
            
        return {
            'action': action,
            'reason': reason,
            'raise_range': ranges['raise_range'],
            'call_range': ranges['call_range']
        }
