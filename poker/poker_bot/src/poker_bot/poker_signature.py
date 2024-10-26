import dspy

class PokerSignature(dspy.Signature):
    """Signature for poker decision making"""
    
    hand = dspy.InputField(desc="The player's hole cards", default="")
    table_cards = dspy.InputField(desc="The community cards on the table", default="")
    position = dspy.InputField(desc="Player's position at the table", default="")
    pot_size = dspy.InputField(desc="Current size of the pot", default=0.0)
    stack_size = dspy.InputField(desc="Player's remaining stack", default=0.0)
    opponent_stack = dspy.InputField(desc="Opponent's remaining stack", default=0.0)
    game_type = dspy.InputField(desc="Type of game being played", default="cash")
    opponent_tendency = dspy.InputField(desc="Opponent's playing style and tendencies", default="unknown")
    
    action = dspy.OutputField(desc="Recommended poker action")
    reasoning = dspy.OutputField(desc="Explanation for the recommended action")
