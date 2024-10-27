import os
import subprocess
import time
import webbrowser
from colorama import Fore, Style
import socket

def check_port(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def wait_for_phoenix():
    """Wait for Phoenix to be ready"""
    print(f"{Fore.YELLOW}Waiting for Phoenix to start...{Style.RESET_ALL}")
    while not (check_port(6006) and check_port(4317)):
        print(".", end="", flush=True)
        time.sleep(1)
    print(f"\n{Fore.GREEN}Phoenix is ready!{Style.RESET_ALL}")

def main():
    # Ensure we're in the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Check if Phoenix is already running
    if not check_port(6006):
        print(f"{Fore.YELLOW}Starting Phoenix server...{Style.RESET_ALL}")
        subprocess.Popen([
            "docker", "run", "-d",
            "-p", "6006:6006",
            "-p", "4317:4317",
            "arizephoenix/phoenix:latest"
        ])
        wait_for_phoenix()
    
    # Open Phoenix UI in browser
    phoenix_url = "http://localhost:6006"
    print(f"\n{Fore.GREEN}Opening Phoenix UI at: {Fore.YELLOW}{phoenix_url}{Style.RESET_ALL}")
    webbrowser.open(phoenix_url)
    
    # Start the poker bot application
    print(f"\n{Fore.GREEN}Starting Poker Bot...{Style.RESET_ALL}")
    from poker_bot.src.poker_bot.main import main as poker_main
    poker_main()

if __name__ == "__main__":
    main()
