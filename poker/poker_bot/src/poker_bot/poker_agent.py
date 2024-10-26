import dspy
from poker_bot.poker_signature import PokerSignature
from poker_bot.safety_checks import SafetyChecks

class PokerAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = PokerSignature
        self.safety_checks = SafetyChecks()
        self.state = {}  # Add state dictionary
    
    def state_dict(self):
        """Return serializable state"""
        return {
            'signature': {
                key: str(value) for key, value in vars(self.signature).items()
                if not key.startswith('_')
            },
            'state': self.state
        }
    
    def load_state_dict(self, state_dict):
        """Load state from dictionary"""
        self.state = state_dict.get('state', {})
        # Restore any signature attributes
        sig_state = state_dict.get('signature', {})
        for key, value in sig_state.items():
            setattr(self.signature, key, value)
    
    def forward(self, hand: str, table_cards: str, position: str, pot_size: float,
                stack_size: float, opponent_stack: float, game_type: str, opponent_tendency: str):
        # Create a new instance with the input parameters
        prediction = self.signature(
            hand=hand,
            table_cards=table_cards,
            position=position,
            pot_size=pot_size,
            stack_size=stack_size,
            opponent_stack=opponent_stack,
            game_type=game_type,
            opponent_tendency=opponent_tendency,
            action="fold",  # Default action
            reasoning=""    # Default reasoning
        )
        
        # Apply safety checks
        if not self.safety_checks.verify_action(prediction.action):
            prediction.action = "fold"
            prediction.reasoning += " [Action adjusted due to safety checks]"
        
        return prediction.action, prediction.reasoning
