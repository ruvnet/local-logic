import dspy
from poker_bot.poker_signature import PokerSignature
from poker_bot.safety_checks import SafetyChecks

class PokerAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(PokerSignature)
        self.safety_checks = SafetyChecks()
    
    def forward(self, hand: str, table_cards: str, position: str, pot_size: float,
                stack_size: float, opponent_stack: float, game_type: str, opponent_tendency: str):
        prediction = self.analyzer(
            hand=hand,
            table_cards=table_cards,
            position=position,
            pot_size=pot_size,
            stack_size=stack_size,
            opponent_stack=opponent_stack,
            game_type=game_type,
            opponent_tendency=opponent_tendency
        )
        # Apply safety checks
        if not self.safety_checks.verify_action(prediction.action):
            prediction.action = "fold"
            prediction.reasoning += " [Action adjusted due to safety checks]"
        return prediction.action, prediction.reasoning
