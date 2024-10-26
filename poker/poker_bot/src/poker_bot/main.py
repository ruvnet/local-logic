from poker_bot.poker_assistant import PokerAssistant
from poker_bot.config import OPENAI_API_KEY
from colorama import init, Fore, Back, Style
import re

init(autoreset=True)  # Initialize colorama

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

def get_valid_cards(prompt, num_cards):
    """Get valid card input from user"""
    while True:
        cards = input(f"{Fore.CYAN}{prompt}{Style.RESET_ALL}").strip()
        if not cards and num_cards == 0:  # Allow empty input for table cards
            return ""
        card_pattern = r'^([2-9TJQKA][hdcs]\s*){' + str(num_cards) + r'}$'
        if re.match(card_pattern, cards, re.I):
            return cards.upper()
        print(f"{Fore.RED}Invalid input! Format example: AH KD (for Ace of Hearts, King of Diamonds){Style.RESET_ALL}")

def print_poker_table():
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.YELLOW}🎰 POKER DECISION ASSISTANT 🎰")
    print(f"{Fore.GREEN}{'='*60}\n")

def print_instructions():
    print(f"\n{Fore.YELLOW}📝 CARD FORMAT INSTRUCTIONS:")
    print(f"{Fore.WHITE}Cards should be entered using two characters:")
    print(f"{Fore.CYAN}First character (Rank): {Fore.WHITE}2-9, T(10), J(Jack), Q(Queen), K(King), A(Ace)")
    print(f"{Fore.CYAN}Second character (Suit): {Fore.WHITE}h(♥️), d(♦️), c(♣️), s(♠️)")
    print(f"{Fore.CYAN}Examples: {Fore.WHITE}AH = Ace of Hearts, KD = King of Diamonds, TC = Ten of Clubs")
    print(f"{Fore.CYAN}Multiple cards: {Fore.WHITE}Separate with spaces (e.g., 'AH KD' for Ace of Hearts and King of Diamonds)\n")

def print_position_guide():
    print(f"\n{Fore.YELLOW}🪑 POSITION GUIDE:")
    print(f"{Fore.CYAN}BTN: {Fore.WHITE}Button/Dealer")
    print(f"{Fore.CYAN}SB:  {Fore.WHITE}Small Blind")
    print(f"{Fore.CYAN}BB:  {Fore.WHITE}Big Blind")
    print(f"{Fore.CYAN}UTG: {Fore.WHITE}Under the Gun (First to act)")
    print(f"{Fore.CYAN}MP:  {Fore.WHITE}Middle Position")
    print(f"{Fore.CYAN}CO:  {Fore.WHITE}Cut Off (Before Button)\n")

def main():
    if not OPENAI_API_KEY:
        raise ValueError(f"{Fore.RED}OpenAI API key is not set.{Style.RESET_ALL}")
    
    print_poker_table()
    print_instructions()
    print_position_guide()
    
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

if __name__ == "__main__":
    main()
