from poker_bot.poker_assistant import PokerAssistant
from colorama import Fore, Style
import random
import time

class DemoMode:
    def __init__(self):
        self.poker_assistant = PokerAssistant()
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
        
    def generate_random_hand(self):
        ranks = '23456789TJQKA'
        suits = 'HDCS'
        cards = []
        while len(cards) < 2:
            card = random.choice(ranks) + random.choice(suits)
            if card not in cards:
                cards.append(card)
        return ' '.join(cards)
    
    def generate_random_community_cards(self, num_cards):
        ranks = '23456789TJQKA'
        suits = 'HDCS'
        cards = []
        while len(cards) < num_cards:
            card = random.choice(ranks) + random.choice(suits)
            if card not in cards:
                cards.append(card)
        return ' '.join(cards)
    
    def simulate_game(self, opponent_level='intermediate', num_hands=5):
        opponent = self.opponent_types[opponent_level]
        print(f"\n{Fore.YELLOW}Starting Demo Mode - Playing against {opponent_level.title()} Opponent")
        print(f"{Fore.CYAN}Opponent Style: {opponent['style']}")
        
        positions = ['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO']
        stack_sizes = [1000, 1500, 2000, 2500, 3000]
        
        wins = 0
        for hand_num in range(num_hands):
            print(f"\n{Fore.GREEN}{'='*60}")
            print(f"{Fore.YELLOW}Demo Hand #{hand_num + 1}")
            print(f"{Fore.GREEN}{'='*60}\n")
            
            # Simulate game state
            position = random.choice(positions)
            stack_size = random.choice(stack_sizes)
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
                wins += 1
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
        print(f"{Fore.WHITE}Hands Won: {wins}")
        print(f"{Fore.WHITE}Win Rate: {(wins/num_hands)*100:.1f}%")
        print(f"{Fore.GREEN}{'='*60}\n")
