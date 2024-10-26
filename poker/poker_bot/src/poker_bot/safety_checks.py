import dspy

class SafetyChecks(dspy.Assert):
    def verify_bet_size(self, bet_size: float, pot_size: float, stack_size: float):
        return 0 <= bet_size <= min(pot_size * 3, stack_size)
    
    def verify_action(self, action: str):
        valid_actions = ['fold', 'call', 'raise', 'all-in']
        return action.lower() in valid_actions
