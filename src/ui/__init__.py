"""
Ultra AI Project - User Interface Module

This module provides various user interface components for the Ultra AI system,
including web interface, CLI interface, and dashboard components.
"""

from .web_interface import WebInterface
from .cli_interface import CLIInterface
from .dashboard import Dashboard

__all__ = [
    'WebInterface',
    'CLIInterface', 
    'Dashboard'
]

__version__ = '1.0.0'
__author__ = 'Ultra AI Team'
__description__ = 'User interface components for Ultra AI system'
