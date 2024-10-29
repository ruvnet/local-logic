from poker_bot.poker_assistant import PokerAssistant
from poker_bot.config import OPENAI_API_KEY
from colorama import init, Fore, Back, Style
from poker_bot.phoenix_config import init_phoenix, get_tracer
import re

init(autoreset=True)  # Initialize colorama

# Initialize Phoenix at startup
print(f"\n{Fore.YELLOW}Initializing Phoenix optimization...{Style.RESET_ALL}")
tracer_provider = init_phoenix()
if not tracer_provider:
    print(f"{Fore.RED}Warning: Phoenix initialization failed. Continuing without optimization.{Style.RESET_ALL}")

# Get tracer for this module
tracer = get_tracer(__name__)

def get_continue_choice():
    while True:
        choice = input(f"\n{Fore.YELLOW}Would you like to analyze another hand? (y/n): {Style.RESET_ALL}").lower()
        if choice in ['y', 'n']:
            return choice == 'y'
        print(f"{Fore.RED}Please enter 'y' for yes or 'n' for no.{Style.RESET_ALL}")

def provide_situation_advice(hand, position, pot_size, stack_size, opponent_stack, game_type, opponent_tendency):
    """Provide situational advice based on current game state"""
    with tracer.start_as_current_span("provide_situation_advice") as span:
        span.set_attribute("position", position)
        span.set_attribute("game_type", game_type)
        
        print(f"\n{Fore.YELLOW}💡 SITUATIONAL ADVICE:")
        
        # Stack size considerations
        stack_to_pot = stack_size / pot_size if pot_size > 0 else float('inf')
        if stack_to_pot < 5:
            print(f"{Fore.CYAN}• Stack-to-pot ratio is low ({stack_to_pot:.1f}x). Consider push/fold strategy.")
        
        # Position-based advice
        position_advice = {
            'BTN': "You have position advantage. Consider stealing if opponents show weakness.",
            'SB': "You'll be out of position postflop. Be more selective with hands.",
            'BB': "You have a discount to see flop. Consider defending wider against steals.",
            'UTG': "Playing from early position requires stronger hands. Be cautious.",
            'MP': "Middle position allows more flexibility. Watch players behind you.",
            'CO': "Strong stealing position. Consider raising with medium-strength hands."
        }
        print(f"{Fore.CYAN}• {position_advice.get(position, 'Unknown position')}")
        
        # Stack size relative to opponent
        stack_ratio = stack_size / opponent_stack
        if stack_ratio < 0.5:
            print(f"{Fore.CYAN}• Short-stacked ({stack_ratio:.1f}x opponent). Look for spots to push all-in.")
        elif stack_ratio > 2:
            print(f"{Fore.CYAN}• Deep-stacked ({stack_ratio:.1f}x opponent). Can play more speculative hands.")
        
        # Game type specific advice
        if game_type.lower() == 'tournament':
            print(f"{Fore.CYAN}• Tournament: Consider ICM implications and stack preservation.")
        else:
            print(f"{Fore.CYAN}• Cash game: Focus on +EV decisions without ICM pressure.")
        
        # Opponent-specific adjustments
        if 'aggressive' in opponent_tendency.lower():
            print(f"{Fore.CYAN}• Against aggressive opponent: Consider trapping and calling down lighter.")
        elif 'passive' in opponent_tendency.lower():
            print(f"{Fore.CYAN}• Against passive opponent: Value bet thinner and bluff less.")
        elif 'smart' in opponent_tendency.lower():
            print(f"{Fore.CYAN}• Against skilled opponent: Avoid predictable patterns and mix up play.")

CARD_SUITS = {
    'h': '♥️',
    'd': '♦️',
    'c': '♣️',
    's': '♠️'
}

def format_cards(cards_str):
    """Convert card notation to emoji format"""
    if not cards_str:
        return ""
    cards = cards_str.split()
    formatted = []
    for card in cards:
        if len(card) == 2:
            rank, suit = card[0], card[1].lower()
            formatted.append(f"{rank}{CARD_SUITS.get(suit, suit)}")
    return ' '.join(formatted)

def normalize_card_input(card_str):
    """Normalize card input to uppercase and handle common variations"""
    # Remove extra spaces and convert to uppercase
    card_str = card_str.strip().upper()
    
    # Handle common variations of suit names
    replacements = {
        'HEARTS': 'H', 'HEART': 'H', '♥': 'H', '♥️': 'H',
        'DIAMONDS': 'D', 'DIAMOND': 'D', '♦': 'D', '♦️': 'D',
        'CLUBS': 'C', 'CLUB': 'C', '♣': 'C', '♣️': 'C',
        'SPADES': 'S', 'SPADE': 'S', '♠': 'S', '♠️': 'S'
    }
    
    for old, new in replacements.items():
        card_str = card_str.replace(old, new)
    
    return card_str

def get_valid_cards(prompt, num_cards):
    """Get valid card input from user with more forgiving validation"""
    with tracer.start_as_current_span("get_valid_cards") as span:
        span.set_attribute("num_cards_required", num_cards)
        
        while True:
            try:
                cards_input = input(f"{Fore.CYAN}{prompt}{Style.RESET_ALL}").strip()
                
                # Handle empty input for table cards
                if not cards_input and num_cards == 0:
                    return ""
                
                # Normalize input
                cards_input = normalize_card_input(cards_input)
                
                # Split into individual cards
                cards = cards_input.split()
                
                # Check number of cards
                if num_cards > 0 and len(cards) != num_cards:
                    print(f"{Fore.RED}Please enter exactly {num_cards} cards.{Style.RESET_ALL}")
                    continue
                
                # Validate each card
                valid_cards = []
                valid_ranks = '23456789TJQKA'
                valid_suits = 'HDCS'
                
                for card in cards:
                    # Handle single character input by prompting for suit
                    if len(card) == 1 and card in valid_ranks:
                        suit = input(f"{Fore.YELLOW}Enter suit for {card} (H/D/C/S): {Style.RESET_ALL}").strip().upper()
                        card = card + suit
                    
                    if len(card) != 2:
                        raise ValueError("Each card must be 2 characters")
                    
                    rank, suit = card[0], card[1]
                    
                    if rank not in valid_ranks:
                        raise ValueError(f"Invalid rank: {rank}")
                    if suit not in valid_suits:
                        raise ValueError(f"Invalid suit: {suit}")
                    
                    valid_cards.append(card)
                
                result = ' '.join(valid_cards)
                span.set_attribute("valid_cards", result)
                return result
                
            except ValueError as e:
                span.record_exception(e)
                print(f"{Fore.RED}Invalid input: {str(e)}")
                print(f"Format examples: AH KD (Ace of Hearts, King of Diamonds)")
                print(f"Valid ranks: 2-9, T(10), J, Q, K, A")
                print(f"Valid suits: H(♥️), D(♦️), C(♣️), S(♠️){Style.RESET_ALL}")

def main():
    with tracer.start_as_current_span("poker_main") as main_span:
        if not OPENAI_API_KEY:
            raise ValueError(f"{Fore.RED}OpenAI API key is not set.{Style.RESET_ALL}")
        
        print_poker_table()
        print_instructions()
        print_position_guide()
        
        while True:
            with tracer.start_as_current_span("game_loop") as game_span:
                print(f"{Fore.GREEN}{'='*60}")
                print(f"{Fore.YELLOW}🎮 GAME SETUP")
                print(f"{Fore.GREEN}{'='*60}\n")
                
                # Get input from user with improved prompts
                print(f"{Fore.YELLOW}First, let's get your hole cards:")
                hand = get_valid_cards(f"Enter your two hole cards: ", 2)
                
                print(f"\n{Fore.YELLOW}Now, let's get the community cards (if any):")
                print(f"{Fore.WHITE}Enter 0-5 cards for pre-flop, flop, turn, or river")
                table_cards = get_valid_cards(f"Enter table cards or press Enter if none: ", 0)
                
                print(f"\n{Fore.YELLOW}What's your position at the table?")
                position = input(f"{Fore.CYAN}Enter position (BTN/SB/BB/UTG/MP/CO): {Style.RESET_ALL}").upper()
                
                print(f"\n{Fore.YELLOW}Let's get the money situation:")
                pot_size = float(input(f"{Fore.CYAN}Enter current pot size ($): {Style.RESET_ALL}"))
                stack_size = float(input(f"{Fore.CYAN}Enter your stack size ($): {Style.RESET_ALL}"))
                opponent_stack = float(input(f"{Fore.CYAN}Enter opponent's stack size ($): {Style.RESET_ALL}"))
                
                print(f"\n{Fore.YELLOW}What type of game is this?")
                game_type = input(f"{Fore.CYAN}Enter game type (cash/tournament): {Style.RESET_ALL}").lower()
                
                print(f"\n{Fore.YELLOW}Finally, tell us about your opponent:")
                print(f"{Fore.WHITE}(e.g., aggressive, passive, tight, loose, bluffs often, etc.)")
                opponent_history = input(f"{Fore.CYAN}Describe opponent's playing style: {Style.RESET_ALL}")

                game_span.set_attribute("hand", hand)
                game_span.set_attribute("position", position)
                game_span.set_attribute("game_type", game_type)

                poker_assistant = PokerAssistant()
                result = poker_assistant.get_action(
                    hand=hand,
                    table_cards=table_cards,
                    position=position,
                    pot_size=pot_size,
                    stack_size=stack_size,
                    opponent_stack=opponent_stack,
                    game_type=game_type,
                    opponent_history=opponent_history
                )

                # Display results with formatting
                print(f"\n{Fore.GREEN}{'='*60}")
                print(f"{Fore.YELLOW}📊 POKER ANALYSIS RESULTS 📊")
                print(f"{Fore.GREEN}{'='*60}\n")
                    
                print(f"{Fore.WHITE}Your Hand: {Fore.RED}{format_cards(hand)}")
                print(f"{Fore.WHITE}Table Cards: {Fore.RED}{format_cards(table_cards)}")
                print(f"{Fore.WHITE}Position: {Fore.YELLOW}{position}")
                print(f"{Fore.WHITE}Pot Size: {Fore.GREEN}${pot_size}")
                print(f"{Fore.WHITE}Your Stack: {Fore.GREEN}${stack_size}")
                
                print(f"\n{Fore.YELLOW}🎯 RECOMMENDATION:")
                print(f"{Fore.WHITE}Action: {Fore.GREEN}{result['recommended_action'].upper()}")
                print(f"{Fore.WHITE}Reasoning: {Fore.CYAN}{result['reasoning']}")
                
                print(f"\n{Fore.YELLOW}📈 ANALYSIS:")
                print(f"{Fore.WHITE}Hand Strength: {Fore.MAGENTA}{result['hand_strength']:.2%}")
                print(f"{Fore.WHITE}Hand Type: {Fore.MAGENTA}{result['hand_type']}")
                print(f"{Fore.WHITE}Position Strategy: {Fore.BLUE}{result['position_strategy']}")
                print(f"{Fore.WHITE}Opponent Tendency: {Fore.RED}{result['opponent_tendency']}")
                
                print(f"\n{Fore.GREEN}{'='*60}\n")
                
                # Add situational advice
                provide_situation_advice(
                    hand=hand,
                    position=position,
                    pot_size=pot_size,
                    stack_size=stack_size,
                    opponent_stack=opponent_stack,
                    game_type=game_type,
                    opponent_tendency=opponent_history
                )
            
                # Ask to continue
                if not get_continue_choice():
                    print(f"\n{Fore.YELLOW}Thanks for using Poker Decision Assistant! Good luck at the tables! 🎰{Style.RESET_ALL}")
                    break

if __name__ == "__main__":
    main()
