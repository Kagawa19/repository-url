#!/usr/bin/env python
import debugpy
import subprocess
import sys
import os
import time

# Setup debugpy and wait for connection
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger to attach at 0.0.0.0:5678...")
debugpy.wait_for_client()
print("Debugger attached! Starting your application...")

# Add an explicit breakpoint test
test_var = "This is a test variable"
print("About to hit test breakpoint...")
# This line should trigger a breakpoint if debugger is properly attached
breakpoint()  # This is Python's built-in debugger function
print("If you see this without stopping, the debugger is not properly attached!")

# Run the init script
result = subprocess.call(["bash", "/app/scripts/init-script.sh"])
sys.exit(result)