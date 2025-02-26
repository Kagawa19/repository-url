#!/usr/bin/env python
import debugpy
import subprocess
import sys
import os

# Setup debugpy and wait for connection
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger to attach at 0.0.0.0:5678...")
debugpy.wait_for_client()
print("Debugger attached! Starting your application...")

# Run the init script
result = subprocess.call(["bash", "/app/scripts/init-script.sh"])
sys.exit(result)