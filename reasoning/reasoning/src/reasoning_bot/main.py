import sys
import dspy
import os
import json
from reasoning_bot.reasoning_assistant import ReasoningAssistant
from reasoning_bot.reasoning_agent import ReasoningAgent
from reasoning_bot.safety_checks import SafetyChecks
from colorama import init, Fore, Style

init(autoreset=True)  # Initialize colorama

# Configure DSPy
dspy.configure(
    lm='gpt-4-mini',  # or your preferred model
    temperature=0.7,
    max_tokens=256
)

class ReasoningModule(dspy.Module):
    def __init__(self):
        super().__init__()
        try:
            # Create signature using input_fields and output_fields
            signature = dspy.Signature(
                input_fields=["input"],
                output_fields=["reasoning"]
            )
            
            # Initialize ChainOfThought with signature and instructions
            self.generate_reasoning = dspy.ChainOfThought(
                signature=signature,
                instructions="Provide detailed step-by-step logical analysis using clear, logical reasoning chains."
            )
        except Exception as e:
            print(f"⚠️ Error initializing ReasoningModule: {str(e)}")
            raise
    
    def forward(self, input_query):
        try:
            if not input_query or not isinstance(input_query, str):
                return "Invalid input. Please provide a valid text query."
                
            # Process query with proper field name
            result = self.generate_reasoning(input=input_query)
            
            # Ensure reasoning field exists
            if not hasattr(result, 'reasoning'):
                return "Unable to generate reasoning. Please try a different query."
                
            return result.reasoning
            
        except Exception as e:
            print(f"⚠️ Reasoning error: {str(e)}")
            return "Unable to process reasoning chain. Please try rephrasing your query."

def simulate_mode(assistant, agent):
    print("🤖 Starting Simulation Mode...")
    
    reasoning_module = ReasoningModule()
    test_cases = [
        "Analyze the implications of increasing system complexity",
        "Evaluate the trade-offs between performance and accuracy",
        "Consider the impact of real-time processing requirements",
        "Assess the benefits of parallel processing implementation"
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📊 Test Case {i}:")
        print(f"Input: {test}")
        dspy_result = reasoning_module(test)
        result = assistant.process_query(test)
        print(f"Basic Analysis: {result}")
        print(f"Deep Reasoning: {dspy_result}")
        print("-" * 50)
    
    print("\n✅ Simulation complete!")

def review_mode(assistant):
    print("🔍 Starting Review Mode...")
    
    # Create logs directory if it doesn't exist
    log_dir = "reasoning_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Load existing logs
    log_file = os.path.join(log_dir, "reasoning_history.json")
    try:
        with open(log_file, 'r') as f:
            logs = json.load(f)
    except FileNotFoundError:
        logs = []
    
    if not logs:
        print("No reasoning history found.")
        return
    
    print(f"\nFound {len(logs)} reasoning sessions:")
    for i, log in enumerate(logs, 1):
        print(f"\n{i}. Session from {log['timestamp']}")
        print(f"Query: {log['query']}")
        print(f"Result: {log['result']}")
        print("-" * 50)

def interactive_mode(assistant, agent, safety_checks):
    print("\n🧠 Starting Interactive Reasoning Session...")
    print("Type 'exit' to quit")
    print("\n💡 TIP: Be specific in your queries for better analysis")
    
    reasoning_module = ReasoningModule()
    
    while True:
        try:
            user_input = input("\n🤔 Enter reasoning query: ")
            
            if user_input.lower() == 'exit':
                print("👋 Ending reasoning session...")
                break
                
            if safety_checks.verify_input(user_input):
                print("\n⚡ Processing query...")
                # Use DSPy for reasoning
                dspy_result = reasoning_module(user_input)
                # Process with assistant
                result = assistant.process_query(user_input)
                
                print(f"\n📝 Reasoning Analysis:")
                print(f"🔍 Initial Analysis: {result}")
                print(f"🧠 Deep Reasoning: {dspy_result}")
                print("\n💭 Additional insights available. Type 'more' for detailed analysis.")
            else:
                print("⚠️ Invalid input detected. Please try again.")
                print("💡 TIP: Ensure your query is clear and well-formed")
                
        except KeyboardInterrupt:
            print("\n👋 Ending reasoning session...")
            break
        except Exception as e:
            print(f"⚠️ Error: {str(e)}")

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
    print("🧠 Initializing Reasoning System Components...")
    
    # Initialize core components
    reasoning_assistant = ReasoningAssistant()
    reasoning_agent = ReasoningAgent()
    safety_checks = SafetyChecks()
    
    print("\n📚 REASONING SYSTEM GUIDE:")
    print("Enter queries in natural language to analyze:")
    print("• Logical problems and scenarios")
    print("• Decision analysis requests") 
    print("• Pattern recognition tasks")
    print("• Complex reasoning chains")
    
    print("\n🔍 QUERY EXAMPLES:")
    print("• Analyze the implications of [scenario]")
    print("• Evaluate the relationship between [A] and [B]")
    print("• Consider the logical consequences of [action]")
    print("• Determine the optimal approach for [situation]")
    
    print("\n⚡ REASONING MODES:")
    print("• Deductive: Step-by-step logical analysis")
    print("• Inductive: Pattern-based reasoning")
    print("• Abductive: Best explanation inference")
    print("• Analogical: Comparison-based reasoning")
    
    print("\n============================================================")
    
    # Get mode from command line argument
    mode = sys.argv[1] if len(sys.argv) > 1 else "interactive"
    
    if mode == "interactive":
        interactive_mode(reasoning_assistant, reasoning_agent, safety_checks)
    elif mode == "simulate":
        simulate_mode(reasoning_assistant, reasoning_agent)
    elif mode == "review":
        review_mode(reasoning_assistant)
    else:
        print(f"⚠️ Unknown mode: {mode}")
    
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

        reasoning_assistant = ReasoningAssistant()
        result = reasoning_assistant.process_query(
            f"Context: Game Type={game_type}, Position={position}\n"
            f"Query: Analyze situation with stack={stack_size}, "
            f"opponent stack={opponent_stack}, pot={pot_size}\n"
            f"Opponent style: {opponent_history}"
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
