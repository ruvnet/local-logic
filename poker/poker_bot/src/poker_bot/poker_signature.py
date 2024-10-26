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
