import subprocess
import os

# Gunicorn Settings
bind = "0.0.0.0:8000"
workers = 3  # Adjust based on CPU cores

def on_starting(server):
    """
    Runs once when the Gunicorn master process starts.
    Use this to launch your data collector script.
    """
    server.log.info("Starting data collector...")
    subprocess.Popen(["python", "pabot.py"])
