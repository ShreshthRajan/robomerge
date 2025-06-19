#!/usr/bin/env python3
"""
RoboMerge Live Dashboard Demo
============================

This script demonstrates the complete RoboMerge pipeline with:
- FAST tokenization
- Knowledge Insulation (KI) for π₀.₅ + KI training  
- Real-Time Chunking (RTC) for deployment
- Live Operations Dashboard

Usage:
    python demo_live_dashboard.py

Requirements:
    - pip install -r requirements.txt
    - Sample data in data/ directory (will create synthetic if missing)
"""

import os
import sys
import time
import json
import threading
import webbrowser
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np
import h5py

# Add robomerge to path
sys.path.append(str(Path(__file__).parent))

def create_synthetic_data(output_path: str):
    """Create synthetic test data for demonstration."""
    print("Creating synthetic robot demonstration data...")
    
    # Generate realistic robot trajectory (8 seconds at 50Hz)
    timesteps = 400
    actions = np.zeros((timesteps, 7))
    states = np.zeros((timesteps, 7))
    timestamps = np.arange(timesteps) / 50.0
    
    # Generate smooth, realistic trajectories
    for joint in range(7):
        base_freq = 0.5 + joint * 0.1
        actions[:, joint] = 0.3 * np.sin(timestamps * base_freq) + 0.1 * np.random.normal(0, 0.1, timesteps)
        states[:, joint] = np.cumsum(actions[:, joint]) * 0.02
    
    # Create dummy camera images
    wrist_img = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
    ext_img = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
    
    # Save as HDF5 (DROID format)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('actions', data=actions)
        f.create_dataset('states', data=states)
        f.create_dataset('timestamps', data=timestamps)
        images_group = f.create_group('images')
        images_group.create_dataset('wrist', data=wrist_img)
        images_group.create_dataset('external', data=ext_img)
    
    return output_path

def setup_demo_data():
    """Set up demonstration data."""
    data_dir = Path("data")
    output_dir = Path("output")
    
    # Create directories
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Create synthetic data file
    test_file = output_dir / "demo_episode.h5"
    create_synthetic_data(str(test_file))
    
    print(f"✅ Demo data created: {test_file}")
    return str(test_file)

def initialize_robomerge():
    """Initialize RoboMerge with all features."""
    from robomerge import RoboMerge
    
    print("🤖 Initializing RoboMerge with KI + RTC + Dashboard...")
    
    # Configure all features
    ki_config = {
        'chunk_size': 50,
        'enable_web_data_mixing': True,
        'gradient_isolation_ratio': 0.7,
        'fast_compression_ratio': 8
    }
    
    rtc_config = {
        'chunk_size': 50,
        'overlap_size': 10,
        'max_latency_ms': 300.0,
        'consistency_threshold': 0.1
    }
    
    dashboard_config = {
        'max_history_hours': 24
    }
    
    # Initialize full pipeline
    pipeline = RoboMerge(
        enable_ki=True,
        enable_rtc=True,
        enable_dashboard=True,
        ki_config=ki_config,
        rtc_config=rtc_config,
        dashboard_config=dashboard_config
    )
    
    return pipeline

def simulate_operator_activity(pipeline):
    """Simulate realistic operator activity for dashboard."""
    print("👥 Simulating operator activity...")
    
    # Add operators
    operators = [
        ("OP_001", "Sarah Chen"),
        ("OP_002", "Marcus Rodriguez"), 
        ("OP_003", "Priya Patel"),
        ("OP_004", "James Kim"),
        ("OP_005", "Anna Lopez"),
        ("OP_006", "David Wilson"),
        ("OP_007", "Lisa Zhang"),
        ("OP_008", "Mohammed Ali")
    ]
    
    for op_id, name in operators:
        pipeline.dashboard.add_operator(op_id, name)
    
    # Simulate demonstration results
    import random
    for i in range(150):  # 150 demonstrations
        operator_id = random.choice([op[0] for op in operators])
        success = random.random() > 0.12  # 88% success rate
        quality_score = random.uniform(0.7, 0.95) if success else random.uniform(0.3, 0.7)
        completion_time = random.uniform(180, 900)
        
        result = {
            'success': success,
            'quality_score': quality_score,
            'completion_time': completion_time,
            'task_type': random.choice(['pick_place', 'folding', 'cleaning', 'assembly', 'inspection'])
        }
        
        pipeline.dashboard.update_operator_metrics(operator_id, result)
        
        # Add queue items
        if i % 20 == 0:
            pipeline.dashboard.add_to_queue(
                task_id=f"task_{i}_{int(time.time())}",
                operator_id=operator_id,
                task_type=result['task_type'],
                priority=random.randint(1, 5)
            )
    
    print("✅ Operator simulation complete")
    return pipeline

def test_all_methods(pipeline, test_file):
    """Test all processing methods."""
    print("\n🧪 Testing All Processing Methods...")
    
    try:
        # Test FAST processing
        print("  Testing FAST...")
        fast_batch = pipeline.process_for_fast(test_file)
        print(f"    ✅ FAST: {fast_batch.action_chunks.shape[0]} chunks")
        
        # Test KI processing
        print("  Testing Knowledge Insulation...")
        ki_batch = pipeline.process_for_ki(
            test_file,
            language_instruction="pick up the red object and place it carefully",
            operator_id="OP_001"
        )
        print(f"    ✅ KI: {ki_batch.discrete_tokens.shape[0]} discrete token chunks")
        print(f"    ✅ KI: {ki_batch.continuous_actions.shape[0]} continuous action chunks")
        
        # Test RTC processing
        print("  Testing Real-Time Chunking...")
        rtc_batch = pipeline.process_for_rtc(
            test_file,
            language_instruction="pick up object for real-time deployment",
            simulated_latency_ms=200.0,
            operator_id="OP_002"
        )
        print(f"    ✅ RTC: {rtc_batch.action_chunks.shape[0]} chunks with latency robustness")
        
        # Test latency robustness
        print("  Testing latency robustness...")
        latency_results = pipeline.test_rtc_robustness(rtc_batch, [0, 100, 200, 300, 400])
        best_perf = latency_results['0ms']['expected_performance']
        worst_perf = latency_results['400ms']['expected_performance']
        print(f"    ✅ Performance: {best_perf:.2f} (0ms) → {worst_perf:.2f} (400ms)")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False

def generate_dashboard_data(pipeline):
    """Generate live dashboard data."""
    summary = pipeline.get_dashboard_summary()
    leaderboard = pipeline.dashboard.get_operator_leaderboard(limit=10)
    queue_status = pipeline.dashboard.get_queue_status()
    active_alerts = pipeline.dashboard.get_active_alerts()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "overview": {
            "active_operators": summary['overview']['active_operators'],
            "total_operators": summary['overview']['total_operators'],
            "demonstrations_today": summary['overview']['total_demonstrations_today'],
            "avg_quality": round(summary['overview']['avg_quality_today'], 3),
            "success_rate": round(summary['overview']['success_rate_today'] * 100, 1),
            "processing_queue": queue_status['total_items'],
            "queue_wait_time": queue_status['avg_wait_time_minutes'],
            "active_alerts": summary['alerts']['total_active'],
            "critical_alerts": summary['alerts']['critical'],
            "high_alerts": summary['alerts']['high']
        },
        "operators": [
            {
                "rank": op['rank'],
                "name": op['name'],
                "operator_id": op['operator_id'],
                "performance_score": op['performance_score'],
                "success_rate": op['success_rate'],
                "avg_quality": op['avg_quality'],
                "completed": op['completed'],
                "failed": op['failed'],
                "status": "Active" if op['rank'] <= 6 else "Idle"
            }
            for op in leaderboard
        ],
        "queue": {
            "total_items": queue_status['total_items'],
            "queued": queue_status['queued'],
            "processing": queue_status['processing'],
            "avg_wait_time": queue_status['avg_wait_time_minutes']
        },
        "alerts": [
            {
                "id": alert.get('alert_id', f"alert_{i}"),
                "severity": alert['severity'],
                "message": alert['message'],
                "operator_id": alert.get('operator_id'),
                "timestamp": alert['timestamp'] if isinstance(alert['timestamp'], str) else alert['timestamp'].isoformat()
            }
            for i, alert in enumerate(active_alerts[:10])
        ]
    }

def create_dashboard_html(data):
    """Create the live dashboard HTML."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RoboMerge Live Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
        }}
        .header {{
            background: rgba(15, 23, 42, 0.95);
            border-bottom: 1px solid #334155;
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .logo {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.5rem;
            font-weight: bold;
            color: #06b6d4;
        }}
        .live-indicator {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid #22c55e;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.875rem;
        }}
        .live-dot {{
            width: 8px;
            height: 8px;
            background: #22c55e;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
        .main-container {{ display: flex; min-height: calc(100vh - 80px); }}
        .sidebar {{
            width: 250px;
            background: rgba(30, 41, 59, 0.7);
            border-right: 1px solid #334155;
            padding: 2rem 0;
            backdrop-filter: blur(10px);
        }}
        .nav-item {{
            padding: 1rem 2rem;
            cursor: pointer;
            transition: all 0.2s;
            border-left: 3px solid transparent;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .nav-item:hover {{
            background: rgba(59, 130, 246, 0.1);
            border-left-color: #3b82f6;
        }}
        .nav-item.active {{
            background: rgba(6, 182, 212, 0.1);
            border-left-color: #06b6d4;
            color: #06b6d4;
        }}
        .content {{ flex: 1; padding: 2rem; overflow-y: auto; }}
        .page {{ display: none; }}
        .page.active {{ display: block; }}
        .page-title {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 2rem;
            color: #06b6d4;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid #334155;
            border-radius: 0.75rem;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }}
        .metric-title {{
            color: #94a3b8;
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 1rem;
        }}
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #e2e8f0;
        }}
        .metric-change {{
            font-size: 0.875rem;
            margin-top: 0.5rem;
            color: #64748b;
        }}
        .data-table {{
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid #334155;
            border-radius: 0.75rem;
            overflow: hidden;
            backdrop-filter: blur(10px);
        }}
        .table-header {{
            background: rgba(15, 23, 42, 0.8);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid #334155;
            font-weight: 600;
            color: #06b6d4;
        }}
        .table-row {{
            display: grid;
            grid-template-columns: 3fr 2fr 2fr 2fr 1fr;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid rgba(51, 65, 85, 0.3);
            align-items: center;
            transition: background 0.2s;
        }}
        .table-row:hover {{
            background: rgba(59, 130, 246, 0.05);
        }}
        .status-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
            text-align: center;
            min-width: 70px;
        }}
        .status-active {{
            background: rgba(34, 197, 94, 0.2);
            color: #22c55e;
            border: 1px solid #22c55e;
        }}
        .status-idle {{
            background: rgba(251, 191, 36, 0.2);
            color: #fbbf24;
            border: 1px solid #fbbf24;
        }}
        .chart-container {{
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid #334155;
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
        }}
        .chart-title {{
            color: #06b6d4;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }}
        .refresh-btn {{
            background: rgba(6, 182, 212, 0.1);
            border: 1px solid #06b6d4;
            color: #06b6d4;
            padding: 0.5rem 1rem;
            border-radius: 0.375rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .refresh-btn:hover {{
            background: rgba(6, 182, 212, 0.2);
        }}
        .alert-item {{
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            padding: 1rem;
            background: rgba(30, 41, 59, 0.3);
            border-left: 4px solid #ef4444;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }}
        .alert-critical {{ border-left-color: #dc2626; }}
        .alert-high {{ border-left-color: #ea580c; }}
        .alert-medium {{ border-left-color: #d97706; }}
        .alert-low {{ border-left-color: #65a30d; }}
        .alert-content {{ flex: 1; }}
        .alert-title {{ font-weight: 600; margin-bottom: 0.25rem; }}
        .alert-time {{ font-size: 0.75rem; color: #94a3b8; }}
        .last-updated {{
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            font-size: 0.75rem;
            color: #64748b;
            background: rgba(15, 23, 42, 0.8);
            padding: 0.5rem 1rem;
            border-radius: 0.375rem;
            backdrop-filter: blur(10px);
            border: 1px solid #334155;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">🤖 RoboMerge</div>
        <div class="live-indicator">
            <div class="live-dot"></div>
            <span>Live Dashboard</span>
            <button class="refresh-btn" onclick="refreshData()">Refresh</button>
        </div>
    </div>

    <div class="main-container">
        <div class="sidebar">
            <div class="nav-item active" onclick="showPage('overview')">📊 Overview</div>
            <div class="nav-item" onclick="showPage('operators')">👥 Operators</div>
            <div class="nav-item" onclick="showPage('queue')">📋 Processing Queue</div>
            <div class="nav-item" onclick="showPage('alerts')">🚨 Alerts</div>
        </div>

        <div class="content">
            <div id="overview" class="page active">
                <h1 class="page-title">Operations Overview</h1>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">Active Operators</div>
                        <div class="metric-value" id="active-operators">{data['overview']['active_operators']}</div>
                        <div class="metric-change">of {data['overview']['total_operators']} total</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Demonstrations Today</div>
                        <div class="metric-value" id="demos-today">{data['overview']['demonstrations_today']}</div>
                        <div class="metric-change">Success: {data['overview']['success_rate']}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Avg Quality Score</div>
                        <div class="metric-value" id="avg-quality">{data['overview']['avg_quality']}</div>
                        <div class="metric-change">Target: 0.85+</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Processing Queue</div>
                        <div class="metric-value" id="queue-size">{data['overview']['processing_queue']}</div>
                        <div class="metric-change">Wait: {data['queue']['avg_wait_time']:.1f} min</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Active Alerts</div>
                        <div class="metric-value" id="active-alerts">{data['overview']['active_alerts']}</div>
                        <div class="metric-change">{data['overview']['critical_alerts']} critical</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Quality Trends (Last 24 Hours)</div>
                    <canvas id="qualityChart" width="400" height="200"></canvas>
                </div>
            </div>

            <div id="operators" class="page">
                <h1 class="page-title">Operator Performance</h1>
                <div class="data-table">
                    <div class="table-header">Operator Performance Leaderboard</div>
                    <div class="table-row" style="font-weight: 600; background: rgba(15, 23, 42, 0.5);">
                        <div>Operator</div>
                        <div>Performance Score</div>
                        <div>Success Rate</div>
                        <div>Avg Quality</div>
                        <div>Status</div>
                    </div>
                    {''.join([f'''
                    <div class="table-row">
                        <div>
                            <div style="font-weight: 600;">#{op["rank"]} {op["name"]}</div>
                            <div style="font-size: 0.75rem; color: #94a3b8;">{op["operator_id"]}</div>
                        </div>
                        <div style="font-weight: 600; color: #06b6d4;">{op["performance_score"]:.1f}</div>
                        <div>{op["success_rate"]:.1f}%</div>
                        <div>{op["avg_quality"]:.2f}</div>
                        <div><span class="status-badge {'status-active' if op['status'] == 'Active' else 'status-idle'}">{op["status"]}</span></div>
                    </div>
                    ''' for op in data['operators'][:8]])}
                </div>
            </div>

            <div id="queue" class="page">
                <h1 class="page-title">Processing Queue</h1>
                <div class="metrics-grid" style="grid-template-columns: repeat(4, 1fr);">
                    <div class="metric-card">
                        <div class="metric-title">Total Items</div>
                        <div class="metric-value">{data['queue']['total_items']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Queued</div>
                        <div class="metric-value">{data['queue']['queued']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Processing</div>
                        <div class="metric-value">{data['queue']['processing']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Avg Wait</div>
                        <div class="metric-value">{data['queue']['avg_wait_time']:.1f}</div>
                        <div class="metric-change">minutes</div>
                    </div>
                </div>
            </div>

            <div id="alerts" class="page">
                <h1 class="page-title">System Alerts</h1>
                {''.join([f'''
                <div class="alert-item alert-{alert['severity']}">
                    <div class="alert-content">
                        <div class="alert-title">{alert['message']}</div>
                        <div class="alert-time">{alert['timestamp'][:19].replace('T', ' ')}</div>
                    </div>
                    <button class="refresh-btn">Resolve</button>
                </div>
                ''' for alert in data['alerts'][:5]])}
            </div>
        </div>
    </div>

    <div class="last-updated">Last updated: {data['timestamp'][:19].replace('T', ' ')}</div>

    <script>
        function showPage(pageId) {{
            document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            document.getElementById(pageId).classList.add('active');
            event.target.classList.add('active');
        }}

        function refreshData() {{
            fetch('/api/refresh')
                .then(response => response.json())
                .then(data => {{
                    console.log('Data refreshed:', data);
                    location.reload();
                }})
                .catch(error => console.error('Refresh failed:', error));
        }}

        // Quality Chart
        const ctx = document.getElementById('qualityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
                datasets: [{{
                    label: 'Quality Score',
                    data: [0.82, 0.85, 0.89, 0.86, 0.87, 0.84],
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
                scales: {{
                    y: {{ beginAtZero: false, min: 0.7, max: 1.0, grid: {{ color: 'rgba(51, 65, 85, 0.3)' }}, ticks: {{ color: '#94a3b8' }} }},
                    x: {{ grid: {{ color: 'rgba(51, 65, 85, 0.3)' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});

        // Auto-refresh every 30 seconds
        setInterval(() => {{
            document.querySelector('.last-updated').textContent = 
                'Last updated: ' + new Date().toLocaleString();
        }}, 30000);

        console.log('🚀 RoboMerge Live Dashboard Loaded!');
    </script>
</body>
</html>"""

class DashboardServer(SimpleHTTPRequestHandler):
    def __init__(self, pipeline, *args, **kwargs):
        self.pipeline = pipeline
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/api/dashboard':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            data = generate_dashboard_data(self.pipeline)
            self.wfile.write(json.dumps(data, default=str).encode())
            
        elif self.path == '/api/refresh':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            data = generate_dashboard_data(self.pipeline)
            response = {"status": "success", "data": data, "timestamp": datetime.now().isoformat()}
            self.wfile.write(json.dumps(response, default=str).encode())
            
        elif self.path == '/' or self.path == '/dashboard':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            data = generate_dashboard_data(self.pipeline)
            html = create_dashboard_html(data)
            self.wfile.write(html.encode())
            
        else:
            super().do_GET()

def launch_dashboard(pipeline):
    """Launch the live dashboard server."""
    PORT = 8001
    
    def make_handler(*args, **kwargs):
        return DashboardServer(pipeline, *args, **kwargs)
    
    try:
        with HTTPServer(("", PORT), make_handler) as httpd:
            print(f"\n🚀 RoboMerge Live Dashboard")
            print(f"   📊 URL: http://localhost:{PORT}")
            print(f"   🔌 API: http://localhost:{PORT}/api/dashboard")
            print(f"   ⚡ Real-time data from RoboMerge backend")
            print(f"\n   🎯 Features Demonstrated:")
            print(f"     • FAST tokenization for efficient action representation")
            print(f"     • Knowledge Insulation (KI) for π₀.₅ + KI training")
            print(f"     • Real-Time Chunking (RTC) for deployment")
            print(f"     • Live operator performance tracking")
            print(f"     • Quality monitoring and alerting")
            print(f"     • Processing queue management")
            print(f"\n   Press Ctrl+C to stop")
            
            # Auto-open browser
            webbrowser.open(f'http://localhost:{PORT}')
            
            # Start server
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n🛑 Dashboard stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

def main():
    """Main demonstration function."""
    print("🤖 RoboMerge Live Dashboard Demo")
    print("=" * 50)
    print("Demonstrating Physical Intelligence methodology:")
    print("• FAST tokenization")
    print("• Knowledge Insulation (π₀.₅ + KI)")
    print("• Real-Time Chunking (RTC)")
    print("• Operations Dashboard")
    print("=" * 50)
    
    try:
        # Step 1: Setup demo data
        test_file = setup_demo_data()
        
        # Step 2: Initialize RoboMerge
        pipeline = initialize_robomerge()
        
        # Step 3: Simulate operator activity
        pipeline = simulate_operator_activity(pipeline)
        
        # Step 4: Test all processing methods
        success = test_all_methods(pipeline, test_file)
        
        if not success:
            print("❌ Testing failed - check your installation")
            return
        
        # Step 5: Display pipeline capabilities
        info = pipeline.get_pipeline_info()
        print(f"\n📋 Pipeline Capabilities:")
        print(f"   Version: {info['pipeline_version']}")
        print(f"   KI Enabled: {'✅' if info['features']['ki_enabled'] else '❌'}")
        print(f"   RTC Enabled: {'✅' if info['features']['rtc_enabled'] else '❌'}")
        print(f"   Dashboard Enabled: {'✅' if info['features']['dashboard_enabled'] else '❌'}")
        print(f"   Output Formats: {', '.join(info['output_formats'])}")
        
        # Step 6: Show live dashboard data
        dashboard_summary = pipeline.get_dashboard_summary()
        print(f"\n📊 Live Dashboard Data:")
        print(f"   Operators: {dashboard_summary['overview']['active_operators']}/{dashboard_summary['overview']['total_operators']}")
        print(f"   Demonstrations: {dashboard_summary['overview']['total_demonstrations_today']}")
        print(f"   Success Rate: {dashboard_summary['overview']['success_rate_today']:.1%}")
        print(f"   Queue: {dashboard_summary['queue']['total_items']} items")
        print(f"   Alerts: {dashboard_summary['alerts']['total_active']} active")
        
        # Step 7: Launch dashboard
        print(f"\n🌐 Launching Live Dashboard...")
        launch_dashboard(pipeline)
        
    except KeyboardInterrupt:
        print(f"\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            pipeline.shutdown()
            print("🧹 Pipeline cleaned up successfully")
        except:
            pass

if __name__ == "__main__":
    main()