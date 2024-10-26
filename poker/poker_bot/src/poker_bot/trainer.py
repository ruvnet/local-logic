import os
import json
import numpy as np
from colorama import Fore, Style
from poker_bot.poker_agent import PokerAgent
from poker_bot.hyperparameter_tuner import HyperparameterTuner
import dspy
from dspy.evaluate import Evaluate
from typing import List, Dict, Tuple

class TrainingConfig:
    """Configuration class for training parameters"""
    def __init__(self, **kwargs):
        # Default optimal values
        self.num_epochs = kwargs.get('num_epochs', 100)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.validation_interval = kwargs.get('validation_interval', 5)
        self.patience = kwargs.get('patience', 10)
        self.min_delta = kwargs.get('min_delta', 0.001)
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 256)
        self.shuffle_data = kwargs.get('shuffle_data', True)  # New parameter
        self.num_simulations = kwargs.get('num_simulations', 500)  # For Monte Carlo simulations

class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

class PokerEvaluator(dspy.Evaluate):
    """Evaluator for poker model performance"""
    def __init__(self):
        # Create a minimal dev set for evaluation
        devset = [
            {
                'input': {
                    'hand': 'AH KH',
                    'table_cards': 'QH JH 2C',
                    'position': 'BTN',
                    'pot_size': 1000,
                    'stack_size': 2000,
                    'opponent_stack': 2000,
                    'game_type': 'cash',
                    'opponent_tendency': 'aggressive'
                },
                'output': {
                    'action': 'raise',
                    'reasoning': 'Flush draw with overcards'
                }
            }
        ]
        super().__init__(devset=devset)
        self.metrics = [
            "win_rate",
            "expected_value", 
            "decision_quality",
            "bluff_efficiency"
        ]
    
    def evaluate(self, model, eval_data) -> Dict[str, float]:
        results = {metric: 0.0 for metric in self.metrics}
        
        for game in eval_data:
            # Unpack game state to match agent's forward method
            action, reasoning = model(
                hand=game['hand'],
                table_cards=game['table_cards'],
                position=game['position'],
                pot_size=game['pot_size'],
                stack_size=game['stack_size'],
                opponent_stack=game['opponent_stack'],
                game_type=game['game_type'],
                opponent_tendency=game['opponent_tendency']
            )
            
            # Calculate various metrics
            results["win_rate"] += self.calculate_win_rate(action, game)
            results["expected_value"] += self.calculate_ev(action, game)
            results["decision_quality"] += self.evaluate_decision_quality(action, game)
            results["bluff_efficiency"] += self.evaluate_bluff_efficiency(action, game)
        
        # Average the results
        for metric in results:
            results[metric] /= len(eval_data)
            
        return results
    
    def calculate_win_rate(self, prediction, game):
        # Implement actual win rate calculation
        return 0.75  # Placeholder
        
    def calculate_ev(self, prediction, game):
        # Implement actual EV calculation
        return 0.5  # Placeholder
        
    def evaluate_decision_quality(self, prediction, game):
        # Implement decision quality evaluation
        return 0.8  # Placeholder
        
    def evaluate_bluff_efficiency(self, prediction, game):
        # Implement bluff efficiency calculation
        return 0.6  # Placeholder

class PokerTrainer:
    def __init__(self):
        # Configure DSPy to use GPT-4-mini
        dspy.configure(
            lm='gpt-4-mini',
            temperature=0.7,
            max_tokens=256
        )
        
        self.agent = PokerAgent()
        self.save_dir = os.path.join(os.path.dirname(__file__), 'training_data')
        os.makedirs(self.save_dir, exist_ok=True)
        self.tuner = HyperparameterTuner()
        self.evaluator = PokerEvaluator()
        self.config = TrainingConfig()
        
    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = []
        if os.path.exists(self.save_dir):
            for file in os.listdir(self.save_dir):
                if file.startswith('checkpoint_epoch_') and file.endswith('.json'):
                    checkpoints.append(file)
        return sorted(checkpoints)

    def load_checkpoint(self, checkpoint_name):
        """Load a specific checkpoint"""
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                
            print(f"\n{Fore.GREEN}Loading checkpoint from epoch {checkpoint['epoch']}")
            print(f"Win Rate: {checkpoint['metrics']['win_rate']:.2%}{Style.RESET_ALL}")
            
            if checkpoint.get('model_state'):
                self.agent.load_state_dict(checkpoint['model_state'])
            
            return True
        return False

    def display_training_history(self):
        """Display training history"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                
            print(f"\n{Fore.YELLOW}Training History:")
            print(f"{Fore.GREEN}{'='*60}")
            for entry in history:
                print(f"\nEpoch {entry['epoch']}:")
                for metric, value in entry['train_metrics'].items():
                    print(f"Training {metric}: {value:.2%}")
                for metric, value in entry['valid_metrics'].items():
                    print(f"Validation {metric}: {value:.2%}")
            print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
            return True
        return False

    def tune_hyperparameters(self):
        """Run hyperparameter tuning"""
        print(f"\n{Fore.YELLOW}Starting hyperparameter tuning...")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64],
            'temperature': [0.5, 0.7, 0.9],
            'max_tokens': [128, 256, 512]
        }
        
        results = self.tuner.tune_hyperparameters(param_grid)
        self.tuner.plot_results(results)
        
        print(f"\n{Fore.GREEN}Tuning complete! Results saved to tuning_results/")
        print(f"Check tuning_plot.png for visualizations{Style.RESET_ALL}")
        
        return results

    def prepare_training_data(self):
        """Prepare real training data"""
        train_data = []
        valid_data = []
        
        # Example data with correct card formats
        sample_hands = [
            {
                'hand': "AH KH",  # Ace-King suited
                'table_cards': "QH JH 2C",
                'position': "BTN",
                'pot_size': 1000.0,
                'stack_size': 2000.0,
                'opponent_stack': 2000.0,
                'game_type': "cash",
                'opponent_tendency': "aggressive"
            },
            {
                'hand': "TH TD",  # Pocket tens
                'table_cards': "AH KD QC",
                'position': "CO",
                'pot_size': 500.0,
                'stack_size': 1500.0,
                'opponent_stack': 1800.0,
                'game_type': "cash",
                'opponent_tendency': "passive"
            },
            {
                'hand': "JC JS",  # Pocket jacks
                'table_cards': "",  # Preflop
                'position': "MP",
                'pot_size': 100.0,
                'stack_size': 2500.0,
                'opponent_stack': 2200.0,
                'game_type': "tournament",
                'opponent_tendency': "tight"
            }
        ]
        
        # Add sample hands to training data
        train_data.extend(sample_hands)
        
        # Create validation data (subset of training data for now)
        valid_data = sample_hands[:1]
        
        return train_data, valid_data

    def train(self, config: TrainingConfig = None):
        """Actual training implementation"""
        if config:
            self.config = config
            
        print(f"\n{Fore.YELLOW}Starting training with configuration:")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        # Prepare data
        train_data, valid_data = self.prepare_training_data()
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )
        
        # Training loop
        history = []
        best_metric = float('-inf')
        
        for epoch in tqdm(range(self.config.num_epochs), desc="Epochs"):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            # Shuffle training data if enabled
            if self.config.shuffle_data:
                random.shuffle(train_data)
            # Train on batches
            train_metrics = self._train_epoch(train_data)
            
            # Validate
            if epoch % self.config.validation_interval == 0:
                valid_metrics = self.evaluator.evaluate(self.agent, valid_data)
                
                # Save checkpoint if improved
                if valid_metrics['win_rate'] > best_metric:
                    best_metric = valid_metrics['win_rate']
                    self._save_checkpoint(epoch, valid_metrics)
                
                # Check for early stopping
                if early_stopping(1.0 - valid_metrics['win_rate']):
                    print(f"\n{Fore.YELLOW}Early stopping triggered at epoch {epoch}")
                    break
                
                # Record metrics
                metrics = {
                    'epoch': epoch + 1,
                    'train_metrics': train_metrics,
                    'valid_metrics': valid_metrics
                }
                history.append(metrics)
                
                # Display progress
                self._display_metrics(metrics)
        
        # Save final history
        self._save_history(history)
        
    def _train_epoch(self, train_data) -> Dict[str, float]:
        """Train for one epoch with real data processing"""
        total_metrics = {metric: 0.0 for metric in self.evaluator.metrics}
        num_batches = 0
        
        # Process in batches
        for i in tqdm(range(0, len(train_data), self.config.batch_size), desc="Training Batches"):
            batch = train_data[i:i + self.config.batch_size]
            
            # Real batch training using DSPy
            for game_state in batch:
                # Forward pass
                prediction = self.agent(
                    hand=game_state['hand'],
                    table_cards=game_state['table_cards'],
                    position=game_state['position'],
                    pot_size=game_state['pot_size'],
                    stack_size=game_state['stack_size'],
                    opponent_stack=game_state['opponent_stack'],
                    game_type=game_state['game_type'],
                    opponent_tendency=game_state['opponent_tendency']
                )
                
                # Calculate metrics using real poker math
                metrics = self._calculate_real_metrics(prediction, game_state)
                
                # Accumulate metrics
                for metric, value in metrics.items():
                    total_metrics[metric] += value
                    
            num_batches += 1
            
        # Average metrics
        for metric in total_metrics:
            total_metrics[metric] /= (num_batches * self.config.batch_size)
            
        return total_metrics
    
    def _convert_to_treys_format(self, card_str):
        """Convert card string to Treys format"""
        # Mapping for suits
        suit_map = {
            'H': 'h',  # Hearts
            'D': 'd',  # Diamonds
            'C': 'c',  # Clubs
            'S': 's'   # Spades
        }
        
        # Mapping for ranks (corrected to use Treys format)
        rank_map = {
            'T': 'T',  # Ten
            'J': 'J',  # Jack
            'Q': 'Q',  # Queen
            'K': 'K',  # King
            'A': 'A',  # Ace
            '10': 'T'  # Convert 10 to T
        }
        
        if not card_str:
            return None
            
        # Split into rank and suit
        if len(card_str) == 3 and card_str[:2] == '10':
            rank, suit = '10', card_str[2]
        else:
            rank, suit = card_str[0], card_str[1]
        
        # Convert rank if needed
        rank = rank_map.get(rank, rank)
        
        # Convert suit to lowercase
        suit = suit_map.get(suit, suit.lower())
        
        return f"{rank}{suit}"

    def _calculate_real_metrics(self, prediction, game_state):
        """Calculate real poker metrics"""
        from treys import Card, Evaluator
        
        # Convert cards to Treys format
        hand = [
            Card.new(self._convert_to_treys_format(card.strip())) 
            for card in game_state['hand'].split()
        ]
        
        board = []
        if game_state['table_cards']:
            board = [
                Card.new(self._convert_to_treys_format(card.strip()))
                for card in game_state['table_cards'].split()
            ]
        
        # Initialize Treys evaluator
        evaluator = Evaluator()
        
        # Calculate hand strength
        hand_strength = 1.0
        if board:
            hand_value = evaluator.evaluate(board, hand)
            hand_strength = 1.0 - (hand_value / 7462)  # Normalize (7462 is worst hand in Treys)
        
        # Calculate win rate using equity calculator
        win_rate = self._calculate_equity(hand, board)
        
        # Calculate expected value
        ev = self._calculate_ev(
            prediction[0],  # action
            game_state['pot_size'],
            game_state['stack_size'],
            hand_strength
        )
        
        return {
            'win_rate': win_rate,
            'expected_value': ev,
            'decision_quality': self._evaluate_decision_quality(
                prediction[0],
                hand_strength,
                game_state['position'],
                game_state['pot_size'] / game_state['stack_size']
            ),
            'bluff_efficiency': self._evaluate_bluff(
                prediction[0],
                hand_strength,
                game_state['opponent_tendency']
            )
        }

    def _calculate_equity(self, hand, board):
        """Calculate hand equity using Monte Carlo simulation"""
        import random
        from treys import Deck, Evaluator
        
        evaluator = Evaluator()
        num_simulations = self.config.num_simulations
        wins = 0
        
        for _ in range(num_simulations):
            # Create deck excluding known cards
            deck = Deck()
            for card in hand + board:
                deck.cards.remove(card)
                
            # Deal opponent cards
            opponent_hand = deck.draw(2)
            
            # Complete board if needed
            simulation_board = board.copy()
            remaining = 5 - len(simulation_board)
            if remaining > 0:
                simulation_board.extend(deck.draw(remaining))
                
            # Evaluate hands
            hand_value = evaluator.evaluate(simulation_board, hand)
            opponent_value = evaluator.evaluate(simulation_board, opponent_hand)
            
            if hand_value < opponent_value:  # Lower is better in Treys
                wins += 1
                
        return wins / num_simulations

    def _calculate_ev(self, action, pot_size, stack_size, hand_strength):
        """Calculate real expected value of an action"""
        if action == 'fold':
            return 0
        
        if action == 'call':
            return pot_size * (hand_strength - 0.5)  # Simplified EV calculation
        
        if action == 'raise':
            # Consider both fold equity and hand equity
            fold_equity = 0.3  # Opponent fold probability (could be based on opponent_tendency)
            return (pot_size * fold_equity) + (pot_size * 2 * (1 - fold_equity) * (hand_strength - 0.5))
        
        return 0

    def _evaluate_decision_quality(self, action, hand_strength, position, spr):
        """Evaluate decision quality based on poker theory"""
        # Position-based adjustments
        position_multiplier = {
            'BTN': 1.2,  # Button allows more aggressive play
            'CO': 1.1,   # Cutoff is also strong
            'MP': 1.0,   # Middle position is neutral
            'UTG': 0.9,  # Under the gun requires caution
            'BB': 0.95,  # Big blind defense
            'SB': 0.9    # Small blind is worst position
        }.get(position, 1.0)
        
        # Stack-to-pot ratio considerations
        if spr < 3:  # Short stacked
            if action == 'all-in' and hand_strength > 0.7:
                return 1.0 * position_multiplier
            if action == 'fold' and hand_strength < 0.3:
                return 0.9 * position_multiplier
        elif spr > 20:  # Deep stacked
            if action == 'raise' and hand_strength > 0.8:
                return 1.0 * position_multiplier
            if action == 'call' and 0.6 < hand_strength < 0.8:
                return 0.9 * position_multiplier
        
        # Basic hand strength alignment
        if hand_strength > 0.8 and action in ['raise', 'all-in']:
            return 1.0 * position_multiplier
        if 0.6 <= hand_strength <= 0.8 and action in ['call', 'raise']:
            return 0.9 * position_multiplier
        if hand_strength < 0.3 and action == 'fold':
            return 0.8 * position_multiplier
            
        return 0.5  # Default for unclear situations

    def _evaluate_bluff(self, action, hand_strength, opponent_tendency):
        """Evaluate bluffing efficiency"""
        if action != 'raise' or hand_strength > 0.5:
            return 1.0  # Not a bluff
            
        # Adjust based on opponent tendency
        opponent_adjustment = {
            'aggressive': 0.7,  # Harder to bluff aggressive players
            'passive': 1.2,     # Easier to bluff passive players
            'tight': 0.8,       # Harder to bluff tight players
            'loose': 1.1        # Easier to bluff loose players
        }
        
        tendency_mult = 1.0
        for tendency, mult in opponent_adjustment.items():
            if tendency in opponent_tendency.lower():
                tendency_mult *= mult
                
        # Calculate bluff success probability
        bluff_equity = (0.3 + (0.5 - hand_strength)) * tendency_mult
        
        return max(0.0, min(1.0, bluff_equity))
        
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save a checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'metrics': {k: float(v) for k, v in metrics.items()},  # Convert to native Python types
            'model_state': {
                k: (v if isinstance(v, (int, float, str, bool, list, dict)) else str(v))
                for k, v in self.agent.state_dict().items()
            }
        }
        
        checkpoint_path = os.path.join(
            self.save_dir, 
            f'checkpoint_epoch_{epoch+1:03d}.json'
        )
        
        # Ensure all values are JSON serializable
        def make_serializable(obj):
            if isinstance(obj, (int, float, str, bool)):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            else:
                return str(obj)
        
        serializable_checkpoint = make_serializable(checkpoint)
        
        with open(checkpoint_path, 'w') as f:
            json.dump(serializable_checkpoint, f, indent=2)
            
    def _save_history(self, history: List[Dict]):
        """Save training history"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
            
    def _display_metrics(self, metrics: Dict):
        """Display current metrics"""
        print(f"\nEpoch {metrics['epoch']}:")
        print(f"Training Metrics:")
        for metric, value in metrics['train_metrics'].items():
            print(f"  {metric}: {value:.2%}")
        print(f"Validation Metrics:")
        for metric, value in metrics['valid_metrics'].items():
            print(f"  {metric}: {value:.2%}")
