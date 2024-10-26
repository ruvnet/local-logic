#!/bin/bash
exec python -m interpreter --install-python-dependencies
#!/bin/bash
if [ "$1" = "interactive" ]; then
    python3 -c "import interpreter; interpreter.chat()"
else
    python3 -i -c "import interpreter"
fi
