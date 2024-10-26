import streamlit as st
from interpreter import interpreter
import subprocess
import os

class InterpreterUI:
    def __init__(self):
        self.interpreter = interpreter
        self.interpreter.auto_run = True
        self.setup_streamlit()

    def setup_streamlit(self):
        st.set_page_config(
            page_title="Open Interpreter Enhanced Environment",
            layout="wide"
        )
        st.title("Open Interpreter Enhanced Environment")

        # VNC Display
        st.components.v1.iframe(
            src="http://localhost:5900",
            height=600,
            scrolling=True
        )

        # Command Input
        self.command = st.text_area(
            "Enter your command or code request:",
            height=100,
            key="command_input"
        )

        # Execute Button
        if st.button("Execute"):
            self.execute_command()

        # History Display
        if 'history' not in st.session_state:
            st.session_state.history = []

        self.display_history()

    def execute_command(self):
        if self.command:
            with st.spinner('Processing...'):
                # Add command to history
                st.session_state.history.append({
                    'command': self.command,
                    'status': 'Running'
                })

                try:
                    # Check if the command is a code request
                    if self.command.lower().startswith("code:"):
                        code_request = self.command[5:].strip()
                        response = self.generate_code(code_request)
                    else:
                        # Execute command using Open Interpreter
                        response = self.interpreter.chat(self.command)

                    # Update history with response
                    st.session_state.history[-1]['status'] = 'Complete'
                    st.session_state.history[-1]['response'] = response

                except Exception as e:
                    st.session_state.history[-1]['status'] = 'Failed'
                    st.session_state.history[-1]['response'] = str(e)

    def generate_code(self, code_request):
        """Generate code using Aider based on natural language description."""
        # Use Aider to generate code
        from aider import Aider
        aider = Aider()
        code = aider.generate_code(code_request)
        # Save code to a file (optional)
        with open('generated_code.py', 'w') as f:
            f.write(code)
        return code

    def display_history(self):
        st.subheader("Command History")
        for item in st.session_state.history[::-1]:
            with st.expander(f"{item['command']} ({item['status']})", expanded=False):
                if 'response' in item:
                    st.code(item['response'], language='python')

if __name__ == "__main__":
    app = InterpreterUI()
