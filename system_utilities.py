
"""

System Utilities v3.0

System monitoring, resource management, and utility functions

"""



import asyncio

import logging

import psutil

import subprocess

from pathlib import Path



logger = logging.getLogger(__name__)



class SystemUtilities:

    """Comprehensive system utilities and monitoring"""

    

    def __init__(self):

        self.monitoring = False

        logger.info("System Utilities v3.0 initialized")

    

    async def initialize(self):

        """Initialize system utilities"""

        try:

            logger.info("Initializing system utilities...")

            self.monitoring = True

            logger.info("✅ System Utilities ready")

            return True

        except Exception as e:

            logger.error(f"Failed to initialize system utilities: {e}")

            return False

    

    async def get_system_metrics(self):

        """Get current system metrics"""

        try:

            cpu_usage = psutil.cpu_percent(interval=1)

            memory = psutil.virtual_memory()

            

            # Try to get temperature (may not work on all systems)

            temperature = 45.0

            try:

                result = subprocess.run(['termux-battery-status'], 

                                      capture_output=True, text=True, timeout=5)

                if result.returncode == 0:

                    import json

                    battery_info = json.loads(result.stdout)

                    temperature = battery_info.get('temperature', 45.0)

            except:

                pass

            

            return {

                'cpu_usage': cpu_usage,

                'memory_usage': memory.percent,

                'temperature': temperature,

                'battery_level': 100

            }

        except Exception as e:

            logger.error(f"Failed to get system metrics: {e}")

            return {

                'cpu_usage': 0.0,

                'memory_usage': 0.0,

                'temperature': 45.0,

                'battery_level': 100

            }

