import dspy
import numpy as np

class HandEvaluator(dspy.Module):
    """Evaluate poker hand strength using advanced algorithms"""
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.Function(self.evaluate_hand)
    
    def evaluate_hand(self, hand: str, table_cards: str):
        # Implement a simplified hand strength evaluation
        # In a real-world scenario, integrate a poker hand evaluator library
        combined_cards = hand.split() + table_cards.split()
        hand_strength = self.calculate_hand_strength(combined_cards)
        hand_type = self.determine_hand_type(hand_strength)
        return {'hand_strength': hand_strength, 'hand_type': hand_type}
    
    def calculate_hand_strength(self, cards):
        # Placeholder for hand strength calculation logic
        return np.random.rand()  # Random strength for demonstration
    
    def determine_hand_type(self, strength):
        # Placeholder for determining hand type based on strength
        if strength > 0.9:
            return "Royal Flush"
        elif strength > 0.8:
            return "Straight Flush"
        elif strength > 0.7:
            return "Four of a Kind"
        elif strength > 0.6:
            return "Full House"
        elif strength > 0.5:
            return "Flush"
        elif strength > 0.4:
            return "Straight"
        elif strength > 0.3:
            return "Three of a Kind"
        elif strength > 0.2:
            return "Two Pair"
        elif strength > 0.1:
            return "One Pair"
        else:
            return "High Card"
    
    def forward(self, hand: str, table_cards: str):
        result = self.evaluate(hand=hand, table_cards=table_cards)
        return result['hand_strength'], result['hand_type']
from treys import Card, Evaluator

class HandEvaluator:
    """Evaluates poker hands using the Treys library"""
    
    def __init__(self):
        self.evaluator = Evaluator()
        
    def evaluate_hand(self, hole_cards: str, board_cards: str = "") -> tuple:
        """
        Evaluate hand strength and type
        Returns (hand_strength, hand_type)
        """
        # Convert string format to Treys Card objects
        hole = [Card.new(card.strip()) for card in hole_cards.split()]
        board = [Card.new(card.strip()) for card in board_cards.split()] if board_cards else []
        
        # Get raw score (lower is better in Treys)
        score = self.evaluator.evaluate(board, hole) if board else 7462  # 7462 is worst possible hand
        
        # Convert to percentile strength (0-1, higher is better)
        strength = 1 - (score / 7462)
        
        # Get hand type
        hand_type = self.evaluator.class_to_string(self.evaluator.get_rank_class(score))
        
        return strength, hand_type
