"""
Graph API Endpoints
Serves generated graphs and catalog to frontend
"""

from flask import Flask, jsonify, send_file, abort
from flask_cors import CORS
from pathlib import Path
import json

app = Flask(__name__)
CORS(app)

INSIGHTS_FILE = Path("../graph_insights/master_insights.json")
GRAPHS_FOLDER = Path("../graphs_output")

@app.route('/api/graphs/catalog', methods=['GET'])
def get_graph_catalog():
    """Return catalog of all available graphs"""
    
    if not INSIGHTS_FILE.exists():
        return jsonify({
            "error": "Insights not found",
            "message": "Run insight_extraction.py first"
        }), 404
    
    try:
        with open(INSIGHTS_FILE, 'r') as f:
            data = json.load(f)
        
        response = {
            "metadata": data.get('metadata', {}),
            "graph_catalog": data.get('graph_catalog', [])
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": "Failed to load catalog",
            "message": str(e)
        }), 500

@app.route('/api/graphs/<filename>', methods=['GET'])
def get_graph(filename):
    """Serve a specific graph file"""
    
    graph_path = GRAPHS_FOLDER / filename
    
    if not graph_path.exists():
        abort(404)
    
    if not graph_path.is_file():
        abort(404)
    
    allowed_extensions = {'.html', '.png', '.jpg', '.jpeg', '.svg'}
    if graph_path.suffix.lower() not in allowed_extensions:
        abort(403)
    
    mimetype = {
        '.html': 'text/html',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.svg': 'image/svg+xml'
    }.get(graph_path.suffix.lower(), 'application/octet-stream')
    
    return send_file(graph_path, mimetype=mimetype)

@app.route('/api/insights', methods=['GET'])
def get_insights():
    """Return all insights for the insights panel"""
    
    if not INSIGHTS_FILE.exists():
        return jsonify({
            "error": "Insights not found",
            "message": "Run insight_extraction.py first"
        }), 404
    
    try:
        with open(INSIGHTS_FILE, 'r') as f:
            data = json.load(f)
        
        insights = data.get('insights', [])
        
        formatted_insights = []
        for insight in insights:
            formatted_insights.append({
                "id": insight.get('id'),
                "category": insight.get('category'),
                "insight": insight.get('insight'),
                "importance": insight.get('importance'),
                "task_type": insight.get('task_type'),
                "grounding": insight.get('grounding', {}),
                "supports_graphs": insight.get('supports_graphs', [])
            })
        
        return jsonify({
            "insights": formatted_insights,
            "total": len(formatted_insights),
            "task_type": data.get('metadata', {}).get('task_type')
        })
    
    except Exception as e:
        return jsonify({
            "error": "Failed to load insights",
            "message": str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    
    insights_available = INSIGHTS_FILE.exists()
    graphs_available = GRAPHS_FOLDER.exists() and any(GRAPHS_FOLDER.glob('*.html'))
    
    return jsonify({
        "status": "healthy",
        "insights_available": insights_available,
        "graphs_available": graphs_available,
        "insights_path": str(INSIGHTS_FILE),
        "graphs_path": str(GRAPHS_FOLDER)
    })

if __name__ == '__main__':
    print("="*80)
    print("GRAPH API SERVER")
    print("="*80)
    print(f"Insights: {INSIGHTS_FILE}")
    print(f"Graphs:   {GRAPHS_FOLDER}")
    print("="*80)
    
    app.run(host='0.0.0.0', port=5000, debug=True)