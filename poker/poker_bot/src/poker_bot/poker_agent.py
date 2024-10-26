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

    def finetune(self, inputs, targets):
        """Train the model using DSPy's predictor"""
        try:
            # Create a predictor for poker decisions
            predictor = dspy.Predict(self.signature)
            
            # Store training examples
            self.training_examples = []
            
            # Process each example
            for input_data, target in zip(inputs, targets):
                try:
                    # Format example for DSPy
                    example = {
                        'input': input_data,
                        'output': {
                            'action': target['action'],
                            'reasoning': target['reasoning']
                        }
                    }
                    self.training_examples.append(example)
                    
                    # Train predictor on this example
                    predictor.train(**example)
                    
                except Exception as e:
                    print(f"Error processing example: {str(e)}")
                    continue
            
            # Store trained predictor
            self.predictor = predictor
            self.use_local_model = True
            return True
            
        except Exception as e:
            print(f"Finetune error: {str(e)}")
            return False

    def local_model_predict(self, input_data):
        """Predict using trained DSPy predictor"""
        try:
            if hasattr(self, 'predictor'):
                prediction = self.predictor(input=input_data)
                return prediction.action, prediction.reasoning
            else:
                return self.query_llm(input_data)
        except Exception as e:
            print(f"Local prediction error: {str(e)}")
            return self.query_llm(input_data)
            
    def _calculate_similarity(self, input1, input2):
        """Calculate similarity between two input states"""
        score = 0.0
        total = 0.0
        
        # Position match
        if input1['position'] == input2['position']:
            score += 1.0
        total += 1.0
        
        # Stack sizes similarity
        if abs(input1['stack_size'] - input2['stack_size']) < 1000:
            score += 1.0
        total += 1.0
        
        # Pot size similarity
        if abs(input1['pot_size'] - input2['pot_size']) < 200:
            score += 1.0
        total += 1.0
        
        # Game type match
        if input1['game_type'] == input2['game_type']:
            score += 1.0
        total += 1.0
        
        return score / total if total > 0 else 0.0
