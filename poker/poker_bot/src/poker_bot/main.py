from poker_bot.poker_assistant import PokerAssistant
from poker_bot.config import OPENAI_API_KEY

def main():
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set.")
    
    poker_assistant = PokerAssistant()
    result = poker_assistant.get_action(
        hand="Ah Kh",
        table_cards="Qh Jh Th",
        position="BTN",
        pot_size=100.0,
        stack_size=1000.0,
        opponent_stack=800.0,
        game_type="tournament",
        opponent_history="Opponent has been playing aggressively, frequently bluffing."
    )
    print("Poker Assistant Recommendation:")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
