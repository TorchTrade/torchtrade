import json
import threading
import time
from datetime import datetime
from typing import Dict, Any, List
from collections import deque
import http.server
import socketserver
import webbrowser
from pathlib import Path

class TradingDashboard:
    def __init__(self, port: int = 8000, max_history: int = 500):
        self.port = port
        self.max_history = max_history
        self.data_lock = threading.Lock()
        
        # Store trading data
        self.portfolio_history = deque(maxlen=max_history)
        self.trade_history = deque(maxlen=max_history)
        self.action_history = deque(maxlen=max_history)
        self.reward_history = deque(maxlen=max_history)
        
        # Current stats
        self.current_stats = {
            'total_steps': 0,
            'total_trades': 0,
            'current_portfolio_value': 0,
            'current_cash': 0,
            'current_position': 0,
            'total_return': 0,
            'last_action': 0,
            'last_reward': 0,
            'start_time': None,
            'last_update': None
        }
        
        self.server = None
        self.server_thread = None
        
    def log_step(self, info_dict: Dict[str, Any], reward: float = 0.0):
        """Log a trading step."""
        timestamp = datetime.now().isoformat()
        
        with self.data_lock:
            # Initialize on first step
            if self.current_stats['start_time'] is None:
                self.current_stats['start_time'] = timestamp
                self.current_stats['initial_portfolio_value'] = info_dict.get('portfolio_value', 0)
            
            # Update portfolio history
            self.portfolio_history.append({
                'timestamp': timestamp,
                'portfolio_value': info_dict.get('portfolio_value', 0),
                'cash': info_dict.get('cash', 0),
                'position_value': info_dict.get('position_market_value', 0)
            })
            
            # Update trade history if trade was executed
            if info_dict.get('trade_executed', False):
                self.trade_history.append({
                    'timestamp': timestamp,
                    'side': info_dict.get('trade_side'),
                    'amount': info_dict.get('trade_amount', 0),
                    'success': info_dict.get('trade_success', False),
                    'portfolio_value': info_dict.get('portfolio_value', 0)
                })
            
            # Update action and reward history
            self.action_history.append({
                'timestamp': timestamp,
                'action': info_dict.get('action', 0)
            })
            
            self.reward_history.append({
                'timestamp': timestamp,
                'reward': reward
            })
            
            # Update current stats
            self.current_stats.update({
                'total_steps': self.current_stats['total_steps'] + 1,
                'current_portfolio_value': info_dict.get('portfolio_value', 0),
                'current_cash': info_dict.get('cash', 0),
                'current_position': info_dict.get('position_qty', 0),
                'last_action': info_dict.get('action', 0),
                'last_reward': reward,
                'last_update': timestamp
            })
            
            if info_dict.get('trade_executed', False):
                self.current_stats['total_trades'] += 1
                
            # Calculate total return
            if self.current_stats.get('initial_portfolio_value', 0) > 0:
                self.current_stats['total_return'] = (
                    (self.current_stats['current_portfolio_value'] - 
                     self.current_stats['initial_portfolio_value']) / 
                    self.current_stats['initial_portfolio_value'] * 100
                )
    
    def get_data(self):
        """Get current dashboard data as JSON."""
        with self.data_lock:
            return {
                'portfolio_history': list(self.portfolio_history),
                'trade_history': list(self.trade_history),
                'action_history': list(self.action_history),
                'reward_history': list(self.reward_history),
                'current_stats': self.current_stats.copy()
            }
    
    def start_server(self, auto_open: bool = True):
        """Start the dashboard web server."""
        if self.server is not None:
            print(f"Dashboard already running on http://localhost:{self.port}")
            return
            
        handler = self._create_handler()
        self.server = socketserver.TCPServer(("", self.port), handler)
        
        def serve():
            print(f"Dashboard started on http://localhost:{self.port}")
            self.server.serve_forever()
        
        self.server_thread = threading.Thread(target=serve, daemon=True)
        self.server_thread.start()
        
        # Only auto-open if running locally
        if auto_open and host in ["localhost", "127.0.0.1"]:
            time.sleep(0.5)  # Give server time to start
            webbrowser.open(f"http://localhost:{self.port}")
    
    def stop_server(self):
        """Stop the dashboard web server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            print("Dashboard stopped")
    
    def _create_handler(self):
        dashboard = self
        
        class DashboardHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(dashboard._get_html().encode())
                elif self.path == '/data':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    data = json.dumps(dashboard.get_data())
                    self.wfile.write(data.encode())
                else:
                    self.send_error(404)
            
            def log_message(self, format, *args):
                pass  # Suppress server logs
        
        return DashboardHandler
    
    def _get_html(self):
        """Generate the dashboard HTML."""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Trading Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stat-value { font-size: 24px; font-weight: bold; color: #333; }
        .stat-label { font-size: 14px; color: #666; margin-top: 5px; }
        .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .chart-full { grid-column: 1 / -1; }
        canvas { max-height: 300px; }
        h1, h2 { color: #333; }
        .status { padding: 10px; border-radius: 4px; margin-bottom: 20px; }
        .status.running { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Trading Dashboard</h1>
        <div class="status running" id="status">Status: Running - Last update: <span id="lastUpdate">-</span></div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="portfolioValue">$0</div>
                <div class="stat-label">Portfolio Value</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="totalReturn">0%</div>
                <div class="stat-label">Total Return</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="totalSteps">0</div>
                <div class="stat-label">Total Steps</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="totalTrades">0</div>
                <div class="stat-label">Total Trades</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="currentCash">$0</div>
                <div class="stat-label">Current Cash</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="currentPosition">0</div>
                <div class="stat-label">Current Position</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container chart-full">
                <h2>Portfolio Value Over Time</h2>
                <canvas id="portfolioChart"></canvas>
            </div>
            <div class="chart-container">
                <h2>Actions Over Time</h2>
                <canvas id="actionChart"></canvas>
            </div>
            <div class="chart-container">
                <h2>Rewards Over Time</h2>
                <canvas id="rewardChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let portfolioChart, actionChart, rewardChart;
        
        function initCharts() {
            const portfolioCtx = document.getElementById('portfolioChart').getContext('2d');
            portfolioChart = new Chart(portfolioCtx, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'Portfolio Value', data: [], borderColor: 'rgb(75, 192, 192)', tension: 0.1 }] },
                options: { responsive: true, scales: { y: { beginAtZero: false } } }
            });
            
            const actionCtx = document.getElementById('actionChart').getContext('2d');
            actionChart = new Chart(actionCtx, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'Action', data: [], borderColor: 'rgb(255, 99, 132)', tension: 0.1, stepped: true }] },
                options: { responsive: true, scales: { y: { min: -1.5, max: 1.5 } } }
            });
            
            const rewardCtx = document.getElementById('rewardChart').getContext('2d');
            rewardChart = new Chart(rewardCtx, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'Reward', data: [], borderColor: 'rgb(54, 162, 235)', tension: 0.1 }] },
                options: { responsive: true, scales: { y: { beginAtZero: true } } }
            });
        }
        
        function updateDashboard() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    // Update stats
                    document.getElementById('portfolioValue').textContent = `$${data.current_stats.current_portfolio_value.toFixed(2)}`;
                    document.getElementById('totalReturn').textContent = `${data.current_stats.total_return.toFixed(2)}%`;
                    document.getElementById('totalSteps').textContent = data.current_stats.total_steps;
                    document.getElementById('totalTrades').textContent = data.current_stats.total_trades;
                    document.getElementById('currentCash').textContent = `$${data.current_stats.current_cash.toFixed(2)}`;
                    document.getElementById('currentPosition').textContent = data.current_stats.current_position.toFixed(4);
                    document.getElementById('lastUpdate').textContent = data.current_stats.last_update || '-';
                    
                    // Update charts
                    const portfolioData = data.portfolio_history.slice(-50); // Show last 50 points
                    portfolioChart.data.labels = portfolioData.map(d => new Date(d.timestamp).toLocaleTimeString());
                    portfolioChart.data.datasets[0].data = portfolioData.map(d => d.portfolio_value);
                    portfolioChart.update('none');
                    
                    const actionData = data.action_history.slice(-50);
                    actionChart.data.labels = actionData.map(d => new Date(d.timestamp).toLocaleTimeString());
                    actionChart.data.datasets[0].data = actionData.map(d => d.action);
                    actionChart.update('none');
                    
                    const rewardData = data.reward_history.slice(-50);
                    rewardChart.data.labels = rewardData.map(d => new Date(d.timestamp).toLocaleTimeString());
                    rewardChart.data.datasets[0].data = rewardData.map(d => d.reward);
                    rewardChart.update('none');
                })
                .catch(error => console.error('Error fetching data:', error));
        }
        
        // Initialize
        initCharts();
        updateDashboard();
        
        // Update every 2 seconds
        setInterval(updateDashboard, 2000);
    </script>
</body>
</html>
        '''