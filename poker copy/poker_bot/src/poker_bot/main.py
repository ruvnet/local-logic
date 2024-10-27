from poker_bot.poker_assistant import PokerAssistant
from poker_bot.config import OPENAI_API_KEY
from colorama import init, Fore, Back, Style
import re

init(autoreset=True)  # Initialize colorama

def get_continue_choice():
    while True:
        choice = input(f"\n{Fore.YELLOW}Would you like to analyze another hand? (y/n): {Style.RESET_ALL}").lower()
        if choice in ['y', 'n']:
            return choice == 'y'
        print(f"{Fore.RED}Please enter 'y' for yes or 'n' for no.{Style.RESET_ALL}")

def provide_situation_advice(hand, position, pot_size, stack_size, opponent_stack, game_type, opponent_tendency):
    """Provide situational advice based on current game state"""
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
            
            return ' '.join(valid_cards)
            
        except ValueError as e:
            print(f"{Fore.RED}Invalid input: {str(e)}")
            print(f"Format examples: AH KD (Ace of Hearts, King of Diamonds)")
            print(f"Valid ranks: 2-9, T(10), J, Q, K, A")
            print(f"Valid suits: H(♥️), D(♦️), C(♣️), S(♠️){Style.RESET_ALL}")

def print_poker_table():
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.YELLOW}🎰 POKER DECISION ASSISTANT 🎰")
    print(f"{Fore.GREEN}{'='*60}\n")

def display_main_menu():
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.YELLOW}🎰 POKER AI TRAINING SYSTEM 🎮")
    print(f"{Fore.GREEN}{'='*60}\n")
    
    print(f"{Fore.CYAN}🎯 MAIN MENU:")
    print(f"\n{Fore.YELLOW}1. Training & Analysis")
    print(f"{Fore.WHITE}   🔄 train     - Start new training session")
    print(f"{Fore.WHITE}   📊 tune      - Optimize hyperparameters")
    print(f"{Fore.WHITE}   📈 history   - View training metrics")
    
    print(f"\n{Fore.YELLOW}2. Game Modes")
    print(f"{Fore.WHITE}   🎮 play      - Start poker assistant")
    print(f"{Fore.WHITE}   🤖 demo      - Practice with AI opponent")
    print(f"{Fore.WHITE}   🔍 analyze   - Analyze hand history")
    
    print(f"\n{Fore.YELLOW}3. Model Management")
    print(f"{Fore.WHITE}   💾 save      - Save current model")
    print(f"{Fore.WHITE}   📂 load      - Load saved model")
    print(f"{Fore.WHITE}   📋 list      - Show saved models")
    
    print(f"\n{Fore.YELLOW}4. System")
    print(f"{Fore.WHITE}   ⚙️  config    - Configure settings")
    print(f"{Fore.WHITE}   ❓ help      - Show detailed help")
    print(f"{Fore.WHITE}   🚪 quit      - Exit system")
    
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.CYAN}Enter command: {Style.RESET_ALL}", end='')

def print_instructions():
    print(f"\n{Fore.YELLOW}📝 CARD FORMAT INSTRUCTIONS:")
    print(f"{Fore.WHITE}Enter cards in any of these formats:")
    print(f"{Fore.CYAN}• Single letters/numbers + suit: {Fore.WHITE}AH, KD, 2C")
    print(f"{Fore.CYAN}• Just the rank (we'll ask for suit): {Fore.WHITE}A, K, 2")
    print(f"{Fore.CYAN}• With emoji suits: {Fore.WHITE}A♥️ K♦️")
    print(f"{Fore.CYAN}• Multiple cards: {Fore.WHITE}separate with spaces (AH KD)")
    print(f"\n{Fore.WHITE}Valid ranks: 2-9, T(10), J(Jack), Q(Queen), K(King), A(Ace)")
    print(f"Valid suits: H(♥️), D(♦️), C(♣️), S(♠️)\n")

def print_position_guide():
    print(f"\n{Fore.YELLOW}🪑 POSITION GUIDE:")
    print(f"{Fore.CYAN}BTN: {Fore.WHITE}Button/Dealer")
    print(f"{Fore.CYAN}SB:  {Fore.WHITE}Small Blind")
    print(f"{Fore.CYAN}BB:  {Fore.WHITE}Big Blind")
    print(f"{Fore.CYAN}UTG: {Fore.WHITE}Under the Gun (First to act)")
    print(f"{Fore.CYAN}MP:  {Fore.WHITE}Middle Position")
    print(f"{Fore.CYAN}CO:  {Fore.WHITE}Cut Off (Before Button)\n")

def print_help_menu():
    print(f"\n{Fore.YELLOW}📚 HELP MENU")
    print(f"{Fore.GREEN}{'='*60}")
    print(f"{Fore.CYAN}1. Game Basics")
    print(f"{Fore.WHITE}   - Card formats and input instructions")
    print(f"{Fore.WHITE}   - Position explanations")
    print(f"{Fore.WHITE}   - Basic commands")
    
    print(f"\n{Fore.CYAN}2. Strategy Guide")
    print(f"{Fore.WHITE}   - Position-based strategy")
    print(f"{Fore.WHITE}   - Stack size considerations")
    print(f"{Fore.WHITE}   - Pot odds and implied odds")
    
    print(f"\n{Fore.CYAN}3. Demo Mode")
    print(f"{Fore.WHITE}   - Practice against AI opponents")
    print(f"{Fore.WHITE}   - Different skill levels")
    print(f"{Fore.WHITE}   - Performance analysis")
    
    print(f"\n{Fore.CYAN}4. Training")
    print(f"{Fore.WHITE}   train              - Start new training session")
    print(f"{Fore.WHITE}   tune               - Run hyperparameter tuning")
    print(f"{Fore.WHITE}   load-checkpoint    - Load a previous checkpoint")
    print(f"{Fore.WHITE}   list-checkpoints   - Show available checkpoints")
    print(f"{Fore.WHITE}   training-history   - Show training history")
    print(f"{Fore.WHITE}   resume-training    - Continue training from checkpoint")
    
    print(f"\n{Fore.CYAN}5. Commands")
    print(f"{Fore.WHITE}   help     - Show this menu")
    print(f"{Fore.WHITE}   demo     - Start demo mode")
    print(f"{Fore.WHITE}   play     - Start regular game")
    print(f"{Fore.WHITE}   quit     - Exit the program")
    print(f"{Fore.GREEN}{'='*60}\n")

def handle_command(command):
    if command == "help":
        print_help_menu()
        return True
    elif command == "demo":
        from poker_bot.demo_mode import DemoMode
        demo = DemoMode()
        print(f"\n{Fore.YELLOW}Select opponent level:")
        print(f"{Fore.CYAN}1. Beginner")
        print(f"{Fore.CYAN}2. Intermediate")
        print(f"{Fore.CYAN}3. Expert")
        choice = input(f"\n{Fore.WHITE}Enter choice (1-3): {Style.RESET_ALL}")
        levels = {
            "1": "beginner",
            "2": "intermediate",
            "3": "expert"
        }
        level = levels.get(choice, "intermediate")
        demo.simulate_game(opponent_level=level)
        return True
    elif command == "train":
        from poker_bot.trainer import PokerTrainer, TrainingConfig
        print(f"\n{Fore.YELLOW}Starting new training session...")
        trainer = PokerTrainer()
            
        # Prompt user for initial parameters with defaults
        num_epochs_input = input(f"{Fore.CYAN}Enter number of epochs [{Style.RESET_ALL}100{Fore.CYAN}]: {Style.RESET_ALL}")
        num_epochs = int(num_epochs_input.strip()) if num_epochs_input.strip() else 100

        batch_size_input = input(f"{Fore.CYAN}Enter batch size [{Style.RESET_ALL}32{Fore.CYAN}]: {Style.RESET_ALL}")
        batch_size = int(batch_size_input.strip()) if batch_size_input.strip() else 32

        learning_rate_input = input(f"{Fore.CYAN}Enter learning rate [{Style.RESET_ALL}0.001{Fore.CYAN}]: {Style.RESET_ALL}")
        learning_rate = float(learning_rate_input.strip()) if learning_rate_input.strip() else 0.001

        config = TrainingConfig(num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
        results_dir = trainer.train(config)

        print(f"\n{Fore.CYAN}Training complete! Results saved to: {results_dir}")
        while True:
            print(f"\nNext steps:")
            print(f"1. 'tune' - Run hyperparameter tuning")
            print(f"2. 'play' - Test the trained model")
            print(f"3. 'quit' - Exit the system")
            next_command = input(f"{Fore.CYAN}Enter command: {Style.RESET_ALL}").lower().strip()
            if next_command in ["tune", "play", "quit"]:
                return handle_command(next_command)
            else:
                print(f"{Fore.RED}Invalid command. Please try again.{Style.RESET_ALL}")
    elif command == "tune":
        from poker_bot.trainer import PokerTrainer
        trainer = PokerTrainer()
            
        # Prompt user for initial parameters with defaults
        print(f"\n{Fore.YELLOW}Enter hyperparameter ranges for tuning (leave blank for defaults):")
            
        learning_rates_input = input(f"{Fore.CYAN}Learning rates (comma-separated) [{Style.RESET_ALL}0.001,0.01,0.1{Fore.CYAN}]: {Style.RESET_ALL}")
        learning_rates = [float(lr.strip()) for lr in learning_rates_input.split(',')] if learning_rates_input.strip() else [0.001, 0.01, 0.1]
            
        batch_sizes_input = input(f"{Fore.CYAN}Batch sizes (comma-separated) [{Style.RESET_ALL}16,32,64{Fore.CYAN}]: {Style.RESET_ALL}")
        batch_sizes = [int(bs.strip()) for bs in batch_sizes_input.split(',')] if batch_sizes_input.strip() else [16, 32, 64]
            
        temperatures_input = input(f"{Fore.CYAN}Temperatures (comma-separated) [{Style.RESET_ALL}0.5,0.7,0.9{Fore.CYAN}]: {Style.RESET_ALL}")
        temperatures = [float(temp.strip()) for temp in temperatures_input.split(',')] if temperatures_input.strip() else [0.5, 0.7, 0.9]
            
        num_epochs_input = input(f"{Fore.CYAN}Number of epochs (comma-separated) [{Style.RESET_ALL}5,10{Fore.CYAN}]: {Style.RESET_ALL}")
        num_epochs_list = [int(ne.strip()) for ne in num_epochs_input.split(',')] if num_epochs_input.strip() else [5, 10]
            
        param_grid = {
            'learning_rate': learning_rates,
            'batch_size': batch_sizes,
            'temperature': temperatures,
            'num_epochs': num_epochs_list
        }
            
        try:
            results = trainer.tune_hyperparameters(param_grid)
            print(f"\n{Fore.YELLOW}Hyperparameter tuning complete.")
            print(f"Best parameters: {results['best_params']}")
            print(f"Best score: {results['best_score']:.3f}")
            print(f"\nYou may now 'train' with best parameters, 'play', or 'quit'.")
        except Exception as e:
            print(f"\n{Fore.RED}Error during hyperparameter tuning: {str(e)}{Style.RESET_ALL}")
        return False
    elif command == "list-checkpoints":
        from poker_bot.trainer import PokerTrainer
        trainer = PokerTrainer()
        checkpoints = trainer.list_checkpoints()
        if checkpoints:
            print(f"\n{Fore.YELLOW}Available Checkpoints:")
            print(f"{Fore.GREEN}{'='*60}")
            for idx, checkpoint in enumerate(checkpoints, 1):
                print(f"{Fore.WHITE}{idx}. {checkpoint}")
            print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}No checkpoints found.{Style.RESET_ALL}")
        return True
    elif command == "load-checkpoint":
        from poker_bot.trainer import PokerTrainer
        trainer = PokerTrainer()
        checkpoints = trainer.list_checkpoints()
        
        if not checkpoints:
            print(f"\n{Fore.RED}No checkpoints found.{Style.RESET_ALL}")
            return True
            
        print(f"\n{Fore.YELLOW}Available Checkpoints:")
        for idx, checkpoint in enumerate(checkpoints, 1):
            print(f"{Fore.WHITE}{idx}. {checkpoint}")
            
        try:
            choice = int(input(f"\n{Fore.CYAN}Enter checkpoint number to load: {Style.RESET_ALL}"))
            if 1 <= choice <= len(checkpoints):
                trainer.load_checkpoint(checkpoints[choice-1])
            else:
                print(f"{Fore.RED}Invalid checkpoint number.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
        return True
    elif command == "training-history":
        from poker_bot.trainer import PokerTrainer
        trainer = PokerTrainer()
        if not trainer.display_training_history():
            print(f"\n{Fore.RED}No training history found.{Style.RESET_ALL}")
        return True
    elif command == "resume-training":
        from poker_bot.trainer import PokerTrainer
        trainer = PokerTrainer()
        checkpoints = trainer.list_checkpoints()
        
        if not checkpoints:
            print(f"\n{Fore.RED}No checkpoints found to resume from.{Style.RESET_ALL}")
            return True
            
        print(f"\n{Fore.YELLOW}Available Checkpoints:")
        for idx, checkpoint in enumerate(checkpoints, 1):
            print(f"{Fore.WHITE}{idx}. {checkpoint}")
            
        try:
            choice = int(input(f"\n{Fore.CYAN}Enter checkpoint number to resume from: {Style.RESET_ALL}"))
            if 1 <= choice <= len(checkpoints):
                if trainer.load_checkpoint(checkpoints[choice-1]):
                    print(f"\n{Fore.YELLOW}Resuming training...")
                    trainer.train(num_epochs=10, batch_size=32)
            else:
                print(f"{Fore.RED}Invalid checkpoint number.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
        return True
    elif command == "quit":
        print(f"\n{Fore.YELLOW}Thanks for using Poker Decision Assistant! Good luck at the tables! 🎰{Style.RESET_ALL}")
        return False
    return True

def main():
    if not OPENAI_API_KEY:
        raise ValueError(f"{Fore.RED}OpenAI API key is not set.{Style.RESET_ALL}")
    
    print_poker_table()
    print_instructions()
    print_position_guide()
    
    while True:
        display_main_menu()
        command = input().lower().strip()
        
        if command == "play":
            break
        result = handle_command(command)
        if not result:
            print(f"\n{Fore.YELLOW}Thank you for using the Poker AI Training System!{Style.RESET_ALL}")
            break
    
    while True:
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
