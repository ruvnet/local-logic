import dspy
from typing import Optional

class PokerSignature(dspy.Signature):
    """Signature for poker decision making"""
    
    hand: str = dspy.InputField(desc="The player's hole cards")
    table_cards: str = dspy.InputField(desc="The community cards on the table")
    position: str = dspy.InputField(desc="Player's position at the table")
    pot_size: float = dspy.InputField(desc="Current size of the pot")
    stack_size: float = dspy.InputField(desc="Player's remaining stack")
    opponent_stack: float = dspy.InputField(desc="Opponent's remaining stack")
    game_type: str = dspy.InputField(desc="Type of game being played")
    opponent_tendency: str = dspy.InputField(desc="Opponent's playing style and tendencies")
    
    action: str = dspy.OutputField(desc="Recommended poker action")
    reasoning: str = dspy.OutputField(desc="Explanation for the recommended action")
