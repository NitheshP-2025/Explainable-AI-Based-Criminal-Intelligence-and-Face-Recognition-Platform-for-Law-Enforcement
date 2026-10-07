# app.py
"""
Application Entry Point - Run the Dashboard
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.deployment.dashboard import CriminalDashboard

if __name__ == "__main__":
    dashboard = CriminalDashboard()
    dashboard.run()
