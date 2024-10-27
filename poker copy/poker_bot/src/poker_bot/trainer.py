import os
import json
import time
import numpy as np
from colorama import Fore, Style
from tqdm import tqdm
import random
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

    def _convert_to_treys_format(self, card_str):
        """Convert card string to Treys format"""
        if not card_str:
            return None
            
        # Split into rank and suit
        if len(card_str) == 3 and card_str[:2] == '10':
            rank, suit = '10', card_str[2]
        else:
            rank, suit = card_str[0], card_str[1]
            
        # Convert rank to Treys format
        rank = rank.upper()
        if rank == '10':
            rank = 'T'
            
        # Convert suit to Treys format (lowercase)
        suit = suit.lower()
        
        # Validate
        if rank not in '23456789TJQKA':
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in 'hdcs':
            raise ValueError(f"Invalid suit: {suit}")
            
        return f"{rank}{suit}"
    
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
        """Calculate actual win rate based on hand strength and action"""
        from treys import Card, Evaluator
        
        # Convert cards to Treys format
        hand = [Card.new(self._convert_to_treys_format(card.strip())) 
               for card in game['hand'].split()]
        board = []
        if game['table_cards']:
            board = [Card.new(self._convert_to_treys_format(card.strip())) 
                    for card in game['table_cards'].split()]
            
        evaluator = Evaluator()
        
        # Calculate base hand strength
        if board:
            hand_value = evaluator.evaluate(board, hand)
            hand_strength = 1.0 - (hand_value / 7462)  # Normalize
        else:
            # Preflop hand strength
            hand_strength = self._calculate_preflop_strength(game['hand'])
            
        # Adjust win rate based on action and position
        position_multiplier = {
            'BTN': 1.2, 'CO': 1.15, 'MP': 1.0,
            'UTG': 0.9, 'BB': 0.95, 'SB': 0.85
        }.get(game['position'], 1.0)
        
        action_multiplier = {
            'fold': 0.0,
            'call': 1.0,
            'raise': 1.2,
            'all-in': 1.3
        }.get(prediction.lower(), 1.0)
        
        return hand_strength * position_multiplier * action_multiplier
        
    def calculate_ev(self, prediction, game):
        """Calculate expected value based on pot odds and win rate"""
        pot_size = float(game['pot_size'])
        stack_size = float(game['stack_size'])
        
        # Calculate pot odds
        if prediction.lower() == 'fold':
            return 0.0
            
        win_rate = self.calculate_win_rate(prediction, game)
        
        if prediction.lower() == 'call':
            return (pot_size * win_rate) - (pot_size * (1 - win_rate))
        elif prediction.lower() == 'raise':
            raise_amount = min(pot_size * 2, stack_size)
            return (raise_amount * win_rate) - (raise_amount * (1 - win_rate))
            
        return 0.0
        
    def evaluate_decision_quality(self, prediction, game):
        """Evaluate decision quality based on GTO principles"""
        win_rate = self.calculate_win_rate(prediction, game)
        pot_odds = float(game['pot_size']) / float(game['stack_size'])
        
        # Basic GTO check
        if win_rate > pot_odds and prediction.lower() in ['call', 'raise']:
            return 1.0
        elif win_rate < pot_odds and prediction.lower() == 'fold':
            return 1.0
        elif win_rate > 0.7 and prediction.lower() == 'raise':
            return 1.0
        elif win_rate < 0.3 and prediction.lower() == 'fold':
            return 1.0
            
        return 0.5
        
    def evaluate_bluff_efficiency(self, prediction, game):
        """Calculate bluff efficiency based on fold equity and board texture"""
        win_rate = self.calculate_win_rate(prediction, game)
        
        # Only evaluate bluffs
        if prediction.lower() != 'raise' or win_rate > 0.5:
            return 1.0
            
        # Calculate fold equity based on opponent tendency
        fold_equity = {
            'aggressive': 0.2,
            'passive': 0.4,
            'tight': 0.6,
            'loose': 0.3
        }.get(game['opponent_tendency'].lower(), 0.3)
        
        # Adjust for position
        position_bonus = {
            'BTN': 0.2,
            'CO': 0.15,
            'MP': 0.1,
            'UTG': 0.0,
            'BB': 0.05,
            'SB': 0.0
        }.get(game['position'], 0.0)
        
        return min(1.0, fold_equity + position_bonus + (0.5 - win_rate))
        
    def _calculate_preflop_strength(self, hand):
        """Calculate preflop hand strength"""
        try:
            cards = hand.split()
            if len(cards) != 2:
                return 0.4  # Default for invalid hands
                
            # Extract ranks and suits
            rank1, suit1 = cards[0][0], cards[0][1]
            rank2, suit2 = cards[1][0], cards[1][1]
            suited = suit1 == suit2
            
            # Convert face cards to values
            rank_values = {'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, 
                          '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 
                          '7': 7, '8': 8, '9': 9}
            
            # Use rank_values for all conversions, no direct int() calls
            r1 = rank_values.get(rank1, 0)
            r2 = rank_values.get(rank2, 0)
            
            if r1 == 0 or r2 == 0:
                return 0.4  # Invalid rank
            
            # Pocket pairs
            if r1 == r2:
                if r1 >= 13:  # AA, KK
                    return 0.85
                elif r1 >= 11:  # QQ, JJ
                    return 0.75
                elif r1 >= 9:  # TT, 99
                    return 0.65
                else:
                    return 0.55
                    
            # High cards
            high_card = max(r1, r2)
            low_card = min(r1, r2)
            gap = high_card - low_card
            
            # Premium hands
            if high_card == 14:  # Ace high
                if low_card >= 12:  # AK, AQ
                    return 0.8 if suited else 0.7
                elif low_card >= 10:  # AJ, AT
                    return 0.7 if suited else 0.6
                    
            # Connected cards
            if gap == 1:
                if min(r1, r2) >= 10:  # KQ, QJ, JT
                    return 0.65 if suited else 0.55
                else:
                    return 0.6 if suited else 0.5
                    
            # Default values
            if suited:
                return max(0.45, 0.6 - (gap * 0.05))
            else:
                return max(0.35, 0.5 - (gap * 0.05))
                
        except Exception as e:
            print(f"Error calculating preflop strength: {str(e)}")
            return 0.4  # Default for error cases

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

    def tune_hyperparameters(self, param_grid):
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

    def train_one_epoch(self, train_data):
        """Train the model for one epoch"""
        self._train_epoch(train_data)

    def train(self, config: TrainingConfig = None):
        """Train the model with comprehensive result tracking"""
        if config:
            self.config = config

        # Initialize response cache once
        if not hasattr(self, 'response_cache'):
            self.response_cache = {}

        print(f"\n{Fore.YELLOW}Starting training with configuration:")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")

        train_data, valid_data = self.prepare_training_data()
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )

        history = []
        best_metric = float('-inf')
        training_start = time.time()

        for epoch in tqdm(range(self.config.num_epochs), desc="Epochs"):
            # For the first epoch, cache LLM responses
            if epoch == 0:
                self._train_epoch(train_data)
            else:
                # Use local model for subsequent epochs
                self.agent.use_local_model = True
                self._train_epoch(train_data)

            if epoch % self.config.validation_interval == 0:
                valid_metrics = self.evaluator.evaluate(self.agent, valid_data)

                train_metrics = self._train_epoch(train_data)
                metrics = {
                    'epoch': epoch + 1,
                    'train_metrics': train_metrics,
                    'valid_metrics': valid_metrics,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                history.append(metrics)

                if valid_metrics['win_rate'] > best_metric:
                    best_metric = valid_metrics['win_rate']
                    self._save_checkpoint(epoch, valid_metrics)

                if early_stopping(1.0 - valid_metrics['win_rate']):
                    print(f"\n{Fore.YELLOW}Early stopping triggered at epoch {epoch}")
                    break

                self._display_metrics(metrics)

        training_duration = time.time() - training_start

        summary = {
            'training_duration': f"{training_duration:.2f} seconds",
            'early_stopping_triggered': early_stopping.early_stop,
            'best_win_rate': best_metric,
            'final_epoch': len(history)
        }

        results_dir = self.save_training_results(
            history=history,
            final_metrics=valid_metrics,
            config=self.config,
            summary=summary
        )

        return results_dir
        
    def _train_epoch(self, train_data) -> Dict[str, float]:
        """Train for one epoch with efficient caching and local model training"""
        total_metrics = {metric: 0.0 for metric in self.evaluator.metrics}
        num_batches = 0

        # Cache to store LLM responses
        if not hasattr(self, 'response_cache'):
            self.response_cache = {}

        # Collect inputs and outputs for local training
        inputs = []
        targets = []

        for i in tqdm(range(0, len(train_data), self.config.batch_size), desc="Training Batches"):
            batch = train_data[i:i + self.config.batch_size]

            for game_state in batch:
                # Create a unique key for the game state
                state_key = json.dumps(game_state, sort_keys=True)

                if state_key in self.response_cache:
                    prediction = self.response_cache[state_key]
                else:
                    # Query the LLM and cache the response
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
                    self.response_cache[state_key] = prediction

                # Prepare data for local model training
                inputs.append(game_state)
                targets.append({
                    'action': prediction[0],
                    'reasoning': prediction[1]
                })

            num_batches += 1

        # Train local model using DSPy fine-tuning
        self.agent.finetune(inputs, targets)

        # Optionally, compute metrics using local model predictions
        for input_data, target in zip(inputs, targets):
            local_prediction = self.agent.local_model_predict(input_data)
            metrics = self._calculate_real_metrics(local_prediction, input_data)
            for metric, value in metrics.items():
                total_metrics[metric] += value

        # Average metrics
        for metric in total_metrics:
            total_metrics[metric] /= (num_batches * self.config.batch_size)

        return total_metrics
    
    def _convert_to_treys_format(self, card_str):
        """Convert card string to Treys format"""
        if not card_str:
            return None
            
        # Split into rank and suit
        if len(card_str) == 3 and card_str[:2] == '10':
            rank, suit = '10', card_str[2]
        else:
            rank, suit = card_str[0], card_str[1]
            
        # Convert rank to Treys format
        rank = rank.upper()
        if rank == '10':
            rank = 'T'
            
        # Convert suit to Treys format (lowercase)
        suit = suit.lower()
        
        # Validate
        if rank not in '23456789TJQKA':
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in 'hdcs':
            raise ValueError(f"Invalid suit: {suit}")
            
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

    def save_training_results(self, history, final_metrics, config, summary):
        """Save comprehensive training results and analysis"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_dir = os.path.join(os.path.dirname(__file__), 'training_results')
        session_dir = os.path.join(results_dir, f'training_session_{timestamp}')
        os.makedirs(session_dir, exist_ok=True)
        
        training_data = {
            'config': vars(config),
            'history': history,
            'final_metrics': final_metrics,
            'summary': summary,
            'timestamp': timestamp
        }
        
        with open(os.path.join(session_dir, 'training_results.json'), 'w') as f:
            json.dump(training_data, f, indent=2)
        
        analysis = self._generate_analysis(history, final_metrics)
        with open(os.path.join(session_dir, 'analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)
        
        report = (
            f"\nTraining Session Summary ({timestamp})\n"
            f"{'='*50}\n"
            f"Duration: {analysis['duration']}\n"
            f"Final Win Rate: {final_metrics['win_rate']:.2%}\n"
            f"Best Epoch: {analysis['best_epoch']}\n"
            f"Convergence Rate: {analysis['convergence_rate']:.2%}\n\n"
            f"Key Metrics:\n"
            f"- Decision Quality: {final_metrics['decision_quality']:.2%}\n"
            f"- Bluff Efficiency: {final_metrics['bluff_efficiency']:.2%}\n"
            f"- Expected Value: {final_metrics['expected_value']:.2%}\n\n"
            f"Training Configuration:\n"
            f"- Epochs: {config.num_epochs}\n"
            f"- Batch Size: {config.batch_size}\n"
            f"- Learning Rate: {config.learning_rate}\n"
            f"- Temperature: {config.temperature}\n"
        )
        
        with open(os.path.join(session_dir, 'summary.txt'), 'w') as f:
            f.write(report)
        
        print(f"\n{Fore.GREEN}Training results saved to: {session_dir}")
        print(f"{Fore.YELLOW}{report}{Style.RESET_ALL}")
        
        return session_dir

    def _generate_analysis(self, history, final_metrics):
        """Generate detailed analysis of training results"""
        win_rates = [epoch['train_metrics']['win_rate'] for epoch in history]
        
        analysis = {
            'duration': f"{len(history)} epochs",
            'best_epoch': history.index(max(history, key=lambda x: x['train_metrics']['win_rate'])) + 1,
            'convergence_rate': (win_rates[-1] - win_rates[0]) / len(history),
            'metrics_progression': {
                'win_rate': self._calculate_progression(history, 'win_rate'),
                'decision_quality': self._calculate_progression(history, 'decision_quality'),
                'bluff_efficiency': self._calculate_progression(history, 'bluff_efficiency'),
                'expected_value': self._calculate_progression(history, 'expected_value')
            },
            'performance_summary': {
                'early_stage': self._calculate_stage_metrics(history[:len(history)//3]),
                'mid_stage': self._calculate_stage_metrics(history[len(history)//3:2*len(history)//3]),
                'late_stage': self._calculate_stage_metrics(history[2*len(history)//3:])
            }
        }
        
        return analysis

    def _calculate_progression(self, history, metric):
        """Calculate progression of a specific metric"""
        values = [epoch['train_metrics'][metric] for epoch in history]
        return {
            'start': values[0],
            'end': values[-1],
            'min': min(values),
            'max': max(values),
            'improvement': values[-1] - values[0]
        }

    def _calculate_stage_metrics(self, stage_history):
        """Calculate average metrics for a training stage with safety checks"""
        metrics = {}
        if not stage_history:  # Handle empty stage
            return {metric: 0.0 for metric in self.evaluator.metrics}
            
        for metric in self.evaluator.metrics:
            try:
                values = [epoch['train_metrics'][metric] for epoch in stage_history]
                metrics[metric] = sum(values) / len(values) if values else 0.0
            except (KeyError, ZeroDivisionError):
                metrics[metric] = 0.0
        return metrics
