import dspy
from poker_bot.poker_signature import PokerSignature
from poker_bot.safety_checks import SafetyChecks

class PokerAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = PokerSignature
        self.safety_checks = SafetyChecks()
        self.state = {}  # Add state dictionary
    
    def state_dict(self):
        """Return serializable state"""
        return {
            'signature': {
                key: str(value) for key, value in vars(self.signature).items()
                if not key.startswith('_')
            },
            'state': self.state
        }
    
    def load_state_dict(self, state_dict):
        """Load state from dictionary"""
        self.state = state_dict.get('state', {})
        # Restore any signature attributes
        sig_state = state_dict.get('signature', {})
        for key, value in sig_state.items():
            setattr(self.signature, key, value)
    
    def __init__(self):
        super().__init__()
        self.signature = PokerSignature
        self.safety_checks = SafetyChecks()
        self.state = {}  # Add state dictionary

        # Initialize a local model placeholder
        self.local_model = None

    def forward(self, hand: str, table_cards: str, position: str, pot_size: float,
                stack_size: float, opponent_stack: float, game_type: str, opponent_tendency: str):
        # Create input dictionary
        input_data = {
            "hand": hand,
            "table_cards": table_cards,
            "position": position,
            "pot_size": pot_size,
            "stack_size": stack_size,
            "opponent_stack": opponent_stack,
            "game_type": game_type,
            "opponent_tendency": opponent_tendency
        }

        # If local model is available, use it
        if self.local_model:
            prediction = self.local_model_predict(input_data)
        else:
            # Query the LLM
            prediction = self.query_llm(input_data)

        # Apply safety checks
        if not self.safety_checks.verify_action(prediction[0]):
            prediction = ("fold", prediction[1] + " [Action adjusted due to safety checks]")

        return prediction

    def query_llm(self, input_data):
        # Use DSPy to query the LLM
        prediction = self.signature(**input_data)
        return prediction.action, prediction.reasoning

    def fine_tune(self, inputs, targets):
        # Use DSPy's fine-tuning capabilities to train a local model
        self.local_model = dspy.fine_tune(self.signature, inputs, targets)

    def local_model_predict(self, input_data):
        # Predict using the fine-tuned local model
        prediction = self.local_model(**input_data)
        return prediction.action, prediction.reasoning
