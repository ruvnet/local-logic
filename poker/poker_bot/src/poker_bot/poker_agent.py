import dspy
from poker_bot.poker_signature import PokerSignature
from poker_bot.safety_checks import SafetyChecks
from opentelemetry import trace

class PokerAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = PokerSignature
        self.safety_checks = SafetyChecks()
        self.state = {}  # Add state dictionary
        self.predictor = dspy.Predict(self.signature)
        self.tracer = trace.get_tracer(__name__)
        self.local_model = None
        self.training_examples = []

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

    def forward(self, hand: str, table_cards: str, position: str, pot_size: float,
                stack_size: float, opponent_stack: float, game_type: str, opponent_tendency: str):
        with self.tracer.start_as_current_span("poker_agent_forward") as span:
            # Add attributes to span
            span.set_attribute("hand", hand)
            span.set_attribute("position", position)
            span.set_attribute("game_type", game_type)
            
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
            if self.local_model and hasattr(self, 'use_local_model') and self.use_local_model:
                with self.tracer.start_as_current_span("local_model_predict") as predict_span:
                    prediction = self.local_model_predict(input_data)
                    predict_span.set_attribute("prediction_source", "local_model")
            else:
                # Query the LLM
                with self.tracer.start_as_current_span("llm_query") as llm_span:
                    prediction = self.query_llm(input_data)
                    llm_span.set_attribute("prediction_source", "llm")

            # Apply safety checks
            with self.tracer.start_as_current_span("safety_checks") as safety_span:
                if not self.safety_checks.verify_action(prediction[0]):
                    prediction = ("fold", prediction[1] + " [Action adjusted due to safety checks]")
                    safety_span.set_attribute("action_adjusted", True)

            span.set_attribute("final_action", prediction[0])
            return prediction

    def query_llm(self, input_data):
        with self.tracer.start_as_current_span("query_llm") as span:
            # Use DSPy to query the LLM
            prediction = self.predictor(self.signature(**input_data))
            span.set_attribute("prediction_action", prediction.action)
            return prediction.action, prediction.reasoning

    def finetune(self, inputs, targets):
        """Train the model on examples"""
        with self.tracer.start_as_current_span("finetune") as span:
            try:
                # Store examples for future predictions
                self.training_examples = []
                for input_data, target in zip(inputs, targets):
                    self.training_examples.append({
                        'input': input_data,
                        'target': {
                            'action': target['action'],
                            'reasoning': target['reasoning']
                        }
                    })
                
                # Train the predictor on examples
                train_data = [
                    (self.signature(**ex['input']), 
                     dspy.Prediction(action=ex['target']['action'], 
                                   reasoning=ex['target']['reasoning']))
                    for ex in self.training_examples
                ]
                
                self.predictor.train(train_data)
                self.use_local_model = True
                
                span.set_attribute("training_examples_count", len(train_data))
                span.set_attribute("training_success", True)
                return True
                
            except Exception as e:
                print(f"Finetune error: {str(e)}")
                span.set_attribute("training_success", False)
                span.record_exception(e)
                return False

    def local_model_predict(self, input_data):
        """Predict using trained predictor"""
        with self.tracer.start_as_current_span("local_model_predict") as span:
            try:
                if not hasattr(self, 'predictor') or not self.training_examples:
                    span.set_attribute("fallback_to_llm", True)
                    return self.query_llm(input_data)
                
                # Use predictor for inference
                prediction = self.predictor(self.signature(**input_data))
                span.set_attribute("prediction_source", "predictor")
                return prediction.action, prediction.reasoning
                
            except Exception as e:
                print(f"Local prediction error: {str(e)}")
                span.record_exception(e)
                span.set_attribute("fallback_to_llm", True)
                return self.query_llm(input_data)
            
    def _calculate_similarity(self, input1, input2):
        """Calculate similarity between two input states"""
        with self.tracer.start_as_current_span("calculate_similarity") as span:
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
            
            similarity = score / total if total > 0 else 0.0
            span.set_attribute("similarity_score", similarity)
            return similarity
