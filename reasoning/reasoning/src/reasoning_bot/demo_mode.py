from poker_bot.poker_assistant import PokerAssistant
from colorama import Fore, Style
import random
import time

class DemoMode:
    def __init__(self):
        self.poker_assistant = PokerAssistant()
        self.positions = ['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO']
        self.stack_sizes = [1000, 1500, 2000, 2500, 3000]
        
        # Add evaluation methods
        def _evaluate_decision(self, decision, scenario, opponent_style):
            """Evaluate decision quality"""
            eval_score = 0.0
            
            # Check position awareness
            if scenario['position'] in decision['reasoning']:
                eval_score += 0.2
                
            # Check opponent adaptation
            if opponent_style.lower() in decision['reasoning'].lower():
                eval_score += 0.2
                
            # Check hand strength consideration
            if 'hand strength' in decision['reasoning'].lower():
                eval_score += 0.2
                
            # Check pot odds awareness
            if 'pot odds' in decision['reasoning'].lower():
                eval_score += 0.2
                
            # Check stack size consideration
            if 'stack' in decision['reasoning'].lower():
                eval_score += 0.2
                
            return eval_score
        
        self.opponent_types = {
            'beginner': {
                'style': 'passive, calls too much, rarely bluffs, plays too many hands',
                'skill_rating': 0.3,
                'aggression': 0.2
            },
            'intermediate': {
                'style': 'balanced player, sometimes aggressive, decent hand selection',
                'skill_rating': 0.6,
                'aggression': 0.5
            },
            'expert': {
                'style': 'highly aggressive, tricky player, strong hand reading ability, strategic',
                'skill_rating': 0.9,
                'aggression': 0.8
            }
        }
        
    def _get_test_scenarios(self):
        """Get predefined test scenarios"""
        return [
            {
                'hand': "AH KH",
                'table_cards': "QH JH 2C",
                'position': "BTN",
                'pot_size': 1000,
                'stack_size': 2000,
                'opponent_stack': 2000,
                'scenario_type': "Flush Draw + Overcards"
            },
            {
                'hand': "JC JD",
                'table_cards': "AH KD QC",
                'position': "MP",
                'pot_size': 800,
                'stack_size': 1500,
                'opponent_stack': 1500,
                'scenario_type': "Pocket Pair vs Overcard Board"
            },
            {
                'hand': "7H 6H",
                'table_cards': "",
                'position': "CO",
                'pot_size': 100,
                'stack_size': 2000,
                'opponent_stack': 2000,
                'scenario_type': "Preflop Speculative Hand"
            },
            {
                'hand': "AS KD",
                'table_cards': "KH 7C 2D",
                'position': "BB",
                'pot_size': 600,
                'stack_size': 1800,
                'opponent_stack': 1600,
                'scenario_type': "Top Pair Top Kicker"
            }
        ]

    def _generate_random_scenarios(self, num_scenarios):
        """Generate diverse random scenarios"""
        scenarios = []
        positions = ['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO']
        stack_sizes = [1000, 1500, 2000, 2500, 3000]
        
        for _ in range(num_scenarios):
            position = random.choice(self.positions)
            stack_size = random.choice(self.stack_sizes)
            pot_size = random.randint(100, int(stack_size/2))
            
            hand = self.generate_random_hand()
            table_cards = self.generate_random_community_cards(random.choice([0, 3, 4, 5]))
            
            scenarios.append({
                'hand': hand,
                'table_cards': table_cards,
                'position': position,
                'pot_size': pot_size,
                'stack_size': stack_size,
                'opponent_stack': stack_size * random.uniform(0.8, 1.2),
                'scenario_type': "Random Scenario"
            })
        
        return scenarios
    
    def generate_random_hand(self):
        """Generate a random 2-card poker hand"""
        ranks = '23456789TJQKA'
        suits = 'HDCS'
        cards = []
        
        # Generate two unique cards
        while len(cards) < 2:
            card = random.choice(ranks) + random.choice(suits)
            if card not in cards:  # Ensure no duplicate cards
                cards.append(card)
                
        return ' '.join(cards)  # Return in format like "AH KD"
    
    def generate_random_community_cards(self, num_cards):
        ranks = '23456789TJQKA'
        suits = 'HDCS'
        cards = []
        while len(cards) < num_cards:
            card = random.choice(ranks) + random.choice(suits)
            if card not in cards:
                cards.append(card)
        return ' '.join(cards)
    
    def simulate_game(self, opponent_level='intermediate', num_hands=5, test_mode=False):
        """Enhanced simulation with testing capabilities"""
        opponent = self.opponent_types[opponent_level]
        print(f"\n{Fore.YELLOW}Starting Demo Mode - Playing against {opponent_level.title()} Opponent")
        print(f"{Fore.CYAN}Opponent Style: {opponent['style']}")
        
        scenarios = self._get_test_scenarios() if test_mode else self._generate_random_scenarios(num_hands)
        results = {
            'wins': 0,
            'decisions': [],
            'scenario_performance': {}
        }
        
        for hand_num, scenario in enumerate(scenarios, 1):
            print(f"\n{Fore.GREEN}{'='*60}")
            print(f"{Fore.YELLOW}Demo Hand #{hand_num + 1}")
            print(f"{Fore.GREEN}{'='*60}\n")
            
            # Simulate game state
            position = random.choice(self.positions)
            stack_size = random.choice(self.stack_sizes)
            pot_size = random.randint(20, 200)
            opponent_stack = stack_size * random.uniform(0.8, 1.2)
            
            hand = self.generate_random_hand()
            table_cards = self.generate_random_community_cards(random.choice([0, 3, 4, 5]))
            
            # Get AI decision
            result = self.poker_assistant.get_action(
                hand=hand,
                table_cards=table_cards,
                position=position,
                pot_size=float(pot_size),
                stack_size=float(stack_size),
                opponent_stack=float(opponent_stack),
                game_type='cash',
                opponent_history=opponent['style']
            )
            
            # Simulate outcome based on opponent skill and decision quality
            decision_quality = random.random()
            if decision_quality > opponent['skill_rating']:
                results['wins'] += 1
                outcome = f"{Fore.GREEN}Won"
            else:
                outcome = f"{Fore.RED}Lost"
            
            # Display hand information
            print(f"{Fore.WHITE}Your Hand: {Fore.RED}{hand}")
            print(f"{Fore.WHITE}Community Cards: {Fore.RED}{table_cards if table_cards else 'None'}")
            print(f"{Fore.WHITE}Position: {Fore.YELLOW}{position}")
            print(f"{Fore.WHITE}Pot Size: ${pot_size}")
            print(f"{Fore.WHITE}Your Stack: ${stack_size}")
            
            print(f"\n{Fore.YELLOW}AI Decision:")
            print(f"{Fore.WHITE}Action: {Fore.GREEN}{result['recommended_action'].upper()}")
            print(f"{Fore.WHITE}Reasoning: {result['reasoning']}")
            
            print(f"\n{Fore.YELLOW}Hand Result: {outcome}")
            
            # Add dramatic pause between hands
            time.sleep(2)
        
        # Show final results
        print(f"\n{Fore.GREEN}{'='*60}")
        print(f"{Fore.YELLOW}Demo Session Results")
        print(f"{Fore.GREEN}{'='*60}")
        print(f"{Fore.WHITE}Hands Played: {num_hands}")
        print(f"{Fore.WHITE}Hands Won: {results['wins']}")
        print(f"{Fore.WHITE}Win Rate: {(results['wins']/num_hands)*100:.1f}%")
        print(f"{Fore.GREEN}{'='*60}\n")
