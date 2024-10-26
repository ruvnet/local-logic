import dspy

class SafetyChecks:
    """Safety checks for poker actions and bets"""
    
    def __init__(self):
        pass
        
    def verify_bet_size(self, bet_size: float, pot_size: float, stack_size: float) -> bool:
        """Verify that bet size is valid"""
        return 0 <= bet_size <= min(pot_size * 3, stack_size)
    
    def verify_action(self, action: str) -> bool:
        """Verify that action is valid"""
        valid_actions = ['fold', 'call', 'raise', 'all-in']
        return action.lower() in valid_actions
