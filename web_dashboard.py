
"""

Ultra AI Web Dashboard v3.0

Advanced web interface with real-time monitoring

"""



import asyncio

import logging

import json

import time

from datetime import datetime

from aiohttp import web

import aiohttp_cors



logger = logging.getLogger(__name__)



class WebDashboard:

    """Web dashboard for Ultra AI system monitoring and control"""

    

    def __init__(self):

        self.app = None

        self.runner = None

        self.site = None

        self.core_engine = None

        self.learning_system = None

        self.interface_manager = None

        self.system_utilities = None

        self.start_time = time.time()

        logger.info("Web Dashboard v3.0 initialized")

    

    def inject_components(self, core_engine, learning_system, interface_manager, system_utilities):

        """Inject system components"""

        self.core_engine = core_engine

        self.learning_system = learning_system

        self.interface_manager = interface_manager

        self.system_utilities = system_utilities

        logger.info("System components injected into dashboard")

    

    async def init_app(self):

        """Initialize web application"""

        self.app = web.Application()

        

        # Setup CORS

        cors = aiohttp_cors.setup(self.app, defaults={

            "*": aiohttp_cors.ResourceOptions(

                allow_credentials=True,

                expose_headers="*",

                allow_headers="*",

                allow_methods="*"

            )

        })

        

        # Add routes

        cors.add(self.app.router.add_get('/', self.dashboard_handler))

        cors.add(self.app.router.add_get('/api/health', self.health_handler))

        cors.add(self.app.router.add_get('/api/status', self.status_handler))

        cors.add(self.app.router.add_get('/manifest.json', self.manifest_handler))

        

        logger.info("Web dashboard app initialized")

    

    async def dashboard_handler(self, request):

        """Main dashboard page"""

        html = '''

<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Ultra AI Dashboard</title>

    <style>

        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }

        .container { max-width: 1200px; margin: 0 auto; }

        .header { text-align: center; margin-bottom: 30px; }

        .card { background: #2a2a2a; border-radius: 8px; padding: 20px; margin: 15px 0; }

        .status-good { color: #00ff88; }

        .status-warning { color: #ffaa00; }

        .status-error { color: #ff4444; }

        .metric { display: inline-block; margin: 10px 20px; }

        .refresh-btn { background: #0066cc; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }

    </style>

</head>

<body>

    <div class="container">

        <div class="header">

            <h1>🤖 Ultra AI Dashboard v3.0</h1>

            <p>Fusion Architecture - Real-time Monitoring</p>

        </div>

        

        <div class="card">

            <h2>System Status</h2>

            <div id="system-status">Loading...</div>

            <button class="refresh-btn" onclick="refreshStatus()">Refresh</button>

        </div>

        

        <div class="card">

            <h2>Performance Metrics</h2>

            <div id="performance-metrics">Loading...</div>

        </div>

        

        <div class="card">

            <h2>System Information</h2>

            <div id="system-info">

                <div class="metric">Uptime: <span id="uptime"></span></div>

                <div class="metric">Version: v3.0 Fusion</div>

                <div class="metric">Platform: Samsung Galaxy S24 Ultra</div>

            </div>

        </div>

    </div>

    

    <script>

        async function refreshStatus() {

            try {

                const response = await fetch('/api/status');

                const data = await response.json();

                

                document.getElementById('system-status').innerHTML = 

                    `<span class="status-${data.health_status?.status || 'error'}">

                        Status: ${data.health_status?.status?.toUpperCase() || 'UNKNOWN'}

                        (Score: ${data.health_status?.health_score || 0}/100)

                    </span>`;

                

                const metrics = data.health_status?.metrics || {};

                document.getElementById('performance-metrics').innerHTML = 

                    `<div class="metric">CPU: ${metrics.cpu_usage?.toFixed(1) || 0}%</div>

                     <div class="metric">Memory: ${metrics.memory_usage?.toFixed(1) || 0}%</div>

                     <div class="metric">Temperature: ${metrics.temperature?.toFixed(1) || 0}°C</div>

                     <div class="metric">Battery: ${metrics.battery_level || 100}%</div>`;

                

            } catch (error) {

                console.error('Failed to refresh status:', error);

                document.getElementById('system-status').innerHTML = 

                    '<span class="status-error">ERROR: Unable to fetch status</span>';

            }

        }

        

        function updateUptime() {

            const startTime = new Date().getTime() - (Date.now() % 86400000);

            const uptime = Math.floor((Date.now() - startTime) / 1000);

            const hours = Math.floor(uptime / 3600);

            const minutes = Math.floor((uptime % 3600) / 60);

            document.getElementById('uptime').textContent = `${hours}h ${minutes}m`;

        }

        

        // Initial load and periodic updates

        refreshStatus();

        updateUptime();

        setInterval(refreshStatus, 10000);

        setInterval(updateUptime, 60000);

    </script>

</body>

</html>'''

        return web.Response(text=html, content_type='text/html')

    

    async def health_handler(self, request):

        """Health check endpoint"""

        if self.core_engine:

            health = await self.core_engine.get_health_status()

            return web.json_response(health)

        return web.json_response({'status': 'error', 'message': 'Core engine not available'})

    

    async def status_handler(self, request):

        """System status endpoint"""

        status = {

            'timestamp': datetime.now().isoformat(),

            'uptime': time.time() - self.start_time,

            'components': {

                'core_engine': bool(self.core_engine),

                'learning_system': bool(self.learning_system),

                'interface_manager': bool(self.interface_manager),

                'system_utilities': bool(self.system_utilities)

            }

        }

        

        if self.core_engine:

            status['health_status'] = await self.core_engine.get_health_status()

        

        return web.json_response(status)

    

    async def manifest_handler(self, request):

        """PWA manifest"""

        manifest = {

            "name": "Ultra AI Dashboard",

            "short_name": "Ultra AI",

            "description": "Ultra AI System Dashboard v3.0",

            "start_url": "/",

            "display": "standalone",

            "background_color": "#1a1a1a",

            "theme_color": "#00ff88"

        }

        return web.json_response(manifest)

    

    async def start(self, host='127.0.0.1', port=8080):

        """Start the dashboard server"""

        try:

            await self.init_app()

            self.runner = web.AppRunner(self.app)

            await self.runner.setup()

            self.site = web.TCPSite(self.runner, host, port)

            await self.site.start()

            logger.info(f"✅ Web Dashboard started on http://{host}:{port}")

        except Exception as e:

            logger.error(f"Failed to start dashboard: {e}")

            raise

    

    async def stop(self):

        """Stop the dashboard server"""

        if self.site:

            await self.site.stop()

        if self.runner:

            await self.runner.cleanup()

        logger.info("Web Dashboard stopped")



async def create_web_dashboard(core_engine, learning_system, interface_manager, system_utilities):

    """Factory function to create configured dashboard"""

    dashboard = WebDashboard()

    dashboard.inject_components(core_engine, learning_system, interface_manager, system_utilities)

    await dashboard.init_app()

    logger.info("✅ Web dashboard created and configured")

    return dashboard

