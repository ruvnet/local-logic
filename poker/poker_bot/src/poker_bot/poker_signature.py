import dspy

class PokerSignature(dspy.Signature):
    """Analyze poker hands and make strategic decisions"""
    hand = dspy.InputField()
    table_cards = dspy.InputField()
    position = dspy.InputField()
    pot_size = dspy.InputField()
    stack_size = dspy.InputField()
    opponent_stack = dspy.InputField()
    game_type = dspy.InputField()
    opponent_tendency = dspy.InputField()
    action = dspy.OutputField()
    reasoning = dspy.OutputField()
import dspy

class PokerSignature(dspy.Signature):
    """Signature for poker decision making"""
    
    hand = dspy.InputField(desc="The player's hole cards")
    table_cards = dspy.InputField(desc="The community cards on the table")
    position = dspy.InputField(desc="Player's position at the table")
    pot_size = dspy.InputField(desc="Current size of the pot")
    stack_size = dspy.InputField(desc="Player's remaining stack")
    opponent_stack = dspy.InputField(desc="Opponent's remaining stack")
    game_type = dspy.InputField(desc="Type of game being played")
    opponent_tendency = dspy.InputField(desc="Opponent's playing style and tendencies")
    
    action = dspy.OutputField(desc="Recommended poker action")
    reasoning = dspy.OutputField(desc="Explanation for the recommended action")
