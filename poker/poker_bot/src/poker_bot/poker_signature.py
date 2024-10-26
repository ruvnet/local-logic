import dspy

class PokerSignature(dspy.Signature):
    """Signature for poker decision making"""
    
    def __init__(self):
        super().__init__()
        
        self.hand = dspy.InputField(desc="The player's hole cards")
        self.table_cards = dspy.InputField(desc="The community cards on the table")
        self.position = dspy.InputField(desc="Player's position at the table")
        self.pot_size = dspy.InputField(desc="Current size of the pot")
        self.stack_size = dspy.InputField(desc="Player's remaining stack")
        self.opponent_stack = dspy.InputField(desc="Opponent's remaining stack")
        self.game_type = dspy.InputField(desc="Type of game being played")
        self.opponent_tendency = dspy.InputField(desc="Opponent's playing style and tendencies")
        
        self.action = dspy.OutputField(desc="Recommended poker action")
        self.reasoning = dspy.OutputField(desc="Explanation for the recommended action")
