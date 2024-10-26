import dspy
from typing import Optional

class PokerSignature(dspy.Signature):
    """Signature for poker decision making"""
    
    hand: str = dspy.InputField(desc="The player's hole cards", default="")
    table_cards: str = dspy.InputField(desc="The community cards on the table", default="")
    position: str = dspy.InputField(desc="Player's position at the table", default="")
    pot_size: float = dspy.InputField(desc="Current size of the pot", default=0.0)
    stack_size: float = dspy.InputField(desc="Player's remaining stack", default=0.0)
    opponent_stack: float = dspy.InputField(desc="Opponent's remaining stack", default=0.0)
    game_type: str = dspy.InputField(desc="Type of game being played", default="")
    opponent_tendency: str = dspy.InputField(desc="Opponent's playing style and tendencies", default="")
    
    action: str = dspy.OutputField(desc="Recommended poker action", default="fold")
    reasoning: str = dspy.OutputField(desc="Explanation for the recommended action", default="")
