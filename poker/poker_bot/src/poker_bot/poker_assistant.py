from poker_bot.poker_agent import PokerAgent
from poker_bot.hand_evaluator import HandEvaluator
from poker_bot.position_strategy import PositionStrategy
from poker_bot.opponent_model import OpponentModel

class PokerAssistant:
    def __init__(self):
        self.agent = PokerAgent()
        self.evaluator = HandEvaluator()
        self.position_strategy = PositionStrategy()
        self.opponent_model = OpponentModel()
                
    def get_action(self, hand: str, table_cards: str, position: str, pot_size: float,
                   stack_size: float, opponent_stack: float, game_type: str, opponent_history: str):
        # Get hand strength evaluation
        strength, hand_type = self.evaluator(hand, table_cards)
        
        # Get opponent tendency
        opponent_tendency = self.opponent_model(opponent_history)
        
        # Get position-based strategy
        position_recommendation = self.position_strategy(position, strength, stack_size, opponent_stack)
        
        # Get final action and reasoning
        action, reasoning = self.agent(
            hand, table_cards, position, pot_size, stack_size, opponent_stack, game_type, opponent_tendency
        )
        
        return {
            'recommended_action': action,
            'reasoning': reasoning,
            'hand_strength': strength,
            'hand_type': hand_type,
            'position_strategy': position_recommendation,
            'opponent_tendency': opponent_tendency
        }
