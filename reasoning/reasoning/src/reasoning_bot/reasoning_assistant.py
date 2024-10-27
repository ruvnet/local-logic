import json
import os
from datetime import datetime

class ReasoningAssistant:
    def __init__(self):
        self.initialized = True
        self.log_dir = "reasoning_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        print("📚 Reasoning Assistant initialized")
        
    def process_query(self, query: str) -> str:
        # Process the query
        result = f"Processed query: {query}"
        
        # Log the query and result
        self._log_query(query, result)
        
        return result
        
    def _log_query(self, query: str, result: str):
        log_file = os.path.join(self.log_dir, "reasoning_history.json")
        
        # Load existing logs
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except FileNotFoundError:
            logs = []
        
        # Add new log entry
        logs.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result": result
        })
        
        # Save updated logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
