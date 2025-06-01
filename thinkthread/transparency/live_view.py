"""
Live Reasoning Visualization - Real-time reasoning visualization
"""

import asyncio
import json
import threading
import webbrowser
from typing import Dict, Any, Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler
# import websocket_server  # Would use actual library in production
import os


class VisualizationServer:
    """WebSocket server for real-time updates"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.server = None
        self.clients = []
        self.running = False
        
    def start(self):
        """Start the WebSocket server"""
        if self.running:
            return
            
        self.server = MockWebsocketServer(self.port + 1)
        self.server.set_fn_new_client(self._new_client)
        self.server.set_fn_client_left(self._client_left)
        
        # Run in separate thread
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = True
        self.thread.start()
        self.running = True
        
    def _run_server(self):
        """Run the server in a thread"""
        self.server.run_forever()
        
    def _new_client(self, client, server):
        """Handle new client connection"""
        self.clients.append(client)
        print(f"New visualization client connected: {client['id']}")
        
    def _client_left(self, client, server):
        """Handle client disconnection"""
        self.clients.remove(client)
        print(f"Visualization client disconnected: {client['id']}")
        
    def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if self.server:
            self.server.send_message_to_all(json.dumps(message))
            
    def stop(self):
        """Stop the server"""
        if self.server:
            self.server.shutdown()
        self.running = False


class LiveReasoningView:
    """Real-time reasoning visualization"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.server = VisualizationServer(port)
        self.http_server = None
        self.active_session = None
        self.nodes = {}
        self.edges = []
        
    def start_session(self, question: str):
        """Start a new visualization session"""
        self.active_session = {
            'id': f"session_{int(time.time())}",
            'question': question,
            'start_time': time.time(),
            'nodes': {},
            'edges': []
        }
        
        # Start servers if not running
        if not self.server.running:
            self.server.start()
            self._start_http_server()
            
        # Send session start event
        self.server.broadcast({
            'type': 'session_start',
            'data': {
                'session_id': self.active_session['id'],
                'question': question
            }
        })
        
    def _start_http_server(self):
        """Start HTTP server for the visualization page"""
        # Create simple HTML page if it doesn't exist
        self._create_visualization_page()
        
        # Start HTTP server in thread
        handler = SimpleHTTPRequestHandler
        self.http_server = HTTPServer(('localhost', self.port), handler)
        
        thread = threading.Thread(target=self.http_server.serve_forever)
        thread.daemon = True
        thread.start()
        
    def _create_visualization_page(self):
        """Create the visualization HTML page"""
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>ThinkThread Reasoning Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
        #header { 
            background: #333; 
            color: white; 
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        #question { font-size: 18px; }
        #stats { font-size: 14px; }
        svg { width: 100%; height: calc(100vh - 60px); }
        .node { cursor: pointer; }
        .node circle { 
            fill: #69b3a2; 
            stroke: #333;
            stroke-width: 2px;
        }
        .node.pruned circle { fill: #ff6b6b; opacity: 0.5; }
        .node text { font-size: 12px; }
        .link { 
            fill: none; 
            stroke: #999;
            stroke-width: 2px;
        }
        .tooltip {
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }
    </style>
</head>
<body>
    <div id="header">
        <div id="question">Waiting for reasoning session...</div>
        <div id="stats">
            <span id="node-count">Nodes: 0</span> | 
            <span id="time">Time: 0s</span>
        </div>
    </div>
    <svg id="visualization"></svg>
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        // Connect to WebSocket
        const ws = new WebSocket('ws://localhost:8081');
        
        // D3.js setup
        const svg = d3.select('#visualization');
        const width = window.innerWidth;
        const height = window.innerHeight - 60;
        
        const simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2));
            
        let nodes = [];
        let links = [];
        
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            handleMessage(msg);
        };
        
        function handleMessage(msg) {
            switch(msg.type) {
                case 'session_start':
                    document.getElementById('question').textContent = msg.data.question;
                    break;
                case 'add_node':
                    addNode(msg.data);
                    break;
                case 'prune_branch':
                    pruneBranch(msg.data.node_id, msg.data.reason);
                    break;
                case 'add_refinement_step':
                    showRefinementStep(msg.data);
                    break;
                case 'add_perspective':
                    addPerspective(msg.data);
                    break;
                case 'add_solution':
                    addSolution(msg.data);
                    break;
            }
            updateStats();
        }
        
        function addNode(nodeData) {
            nodes.push({
                id: nodeData.id,
                content: nodeData.content,
                score: nodeData.score,
                depth: nodeData.depth || 0
            });
            
            if (nodeData.parent) {
                links.push({
                    source: nodeData.parent,
                    target: nodeData.id
                });
            }
            
            updateVisualization();
        }
        
        function updateVisualization() {
            // Update links
            const link = svg.selectAll('.link')
                .data(links, d => d.source + '-' + d.target);
                
            link.enter().append('line')
                .attr('class', 'link');
                
            // Update nodes
            const node = svg.selectAll('.node')
                .data(nodes, d => d.id);
                
            const nodeEnter = node.enter().append('g')
                .attr('class', 'node')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));
                    
            nodeEnter.append('circle')
                .attr('r', d => 10 + d.score * 20);
                
            nodeEnter.append('text')
                .attr('dy', 3)
                .attr('x', 15)
                .text(d => d.content.substring(0, 30) + '...');
                
            // Restart simulation
            simulation.nodes(nodes);
            simulation.force('link').links(links);
            simulation.alpha(1).restart();
        }
        
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        function updateStats() {
            document.getElementById('node-count').textContent = `Nodes: ${nodes.length}`;
        }
        
        // Update positions
        simulation.on('tick', () => {
            svg.selectAll('.link')
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
                
            svg.selectAll('.node')
                .attr('transform', d => `translate(${d.x},${d.y})`);
        });
    </script>
</body>
</html>"""
        
        with open('reasoning_visualization.html', 'w') as f:
            f.write(html_content)
            
    def add_node(self, node_data: Dict[str, Any]):
        """Add a node to the visualization"""
        self.nodes[node_data['id']] = node_data
        self.server.broadcast({
            'type': 'add_node',
            'data': node_data
        })
        
    def prune_branch(self, node_id: str, reason: str):
        """Show branch pruning with explanation"""
        self.server.broadcast({
            'type': 'prune_branch',
            'data': {
                'node_id': node_id,
                'reason': reason
            }
        })
        
    def add_refinement_step(self, step_data: Dict[str, Any]):
        """Add refinement step visualization"""
        self.server.broadcast({
            'type': 'add_refinement_step',
            'data': step_data
        })
        
    def add_perspective(self, perspective_data: Dict[str, Any]):
        """Add perspective for debate mode"""
        self.server.broadcast({
            'type': 'add_perspective',
            'data': perspective_data
        })
        
    def add_exchange(self, exchange_data: Dict[str, Any]):
        """Add debate exchange"""
        self.server.broadcast({
            'type': 'add_exchange',
            'data': exchange_data
        })
        
    def add_solution(self, solution_data: Dict[str, Any]):
        """Add solution for solve mode"""
        self.server.broadcast({
            'type': 'add_solution',
            'data': solution_data
        })
        
    def open_debug_ui(self, session):
        """Open debugging UI for a session"""
        webbrowser.open(f"http://localhost:{self.port}/reasoning_visualization.html")


# Mock websocket_server for demonstration (in real implementation, use actual library)
class MockWebsocketServer:
        def __init__(self, port):
            self.port = port
            self.clients = []
            
        def set_fn_new_client(self, fn):
            self.new_client_fn = fn
            
        def set_fn_client_left(self, fn):
            self.client_left_fn = fn
            
        def run_forever(self):
            # Mock implementation
            import time
            while True:
                time.sleep(1)
                
        def send_message_to_all(self, message):
            # Mock implementation
            pass
            
        def shutdown(self):
            # Mock implementation
            pass