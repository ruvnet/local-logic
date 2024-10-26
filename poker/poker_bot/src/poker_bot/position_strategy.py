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
