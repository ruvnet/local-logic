from poker_bot.poker_agent import PokerAgent

class PokerAssistant:
    def __init__(self):
        self.agent = PokerAgent()
                
    def get_action(self, hand: str, table_cards: str, position: str, pot_size: float,
                   stack_size: float, opponent_stack: float, game_type: str, opponent_history: str):
        # Get final action and reasoning
        action, reasoning = self.agent(
            hand, table_cards, position, pot_size, stack_size, opponent_stack, 
            game_type, opponent_history
        )
        
        return {
            'recommended_action': action,
            'reasoning': reasoning,
            'hand_strength': 0.0,  # Placeholder until HandEvaluator is implemented
            'hand_type': "Unknown",  # Placeholder until HandEvaluator is implemented
            'position_strategy': "Basic",  # Placeholder until PositionStrategy is implemented
            'opponent_tendency': opponent_history  # Using raw history until OpponentModel is implemented
        }
