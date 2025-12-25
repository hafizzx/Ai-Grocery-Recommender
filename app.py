from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model
try:
    from model import GroceryRecommender
    print("✅ Successfully imported GroceryRecommender")
except ImportError as e:
    print(f"❌ Error importing model: {e}")
    print("Creating mock recommender...")
    
    # Mock class for testing
    class GroceryRecommender:
        def __init__(self):
            self.df = None
            self.is_trained = False
        
        def search_products(self, **kwargs):
            return []
        
        def get_categories(self):
            return []
        
        def recommend(self, **kwargs):
            return []
        
        def get_product_stats(self):
            return {}

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Initialize recommender
print("\n" + "="*60)
print("🤖 INITIALIZING AI GROCERY RECOMMENDER")
print("="*60)

try:
    recommender = GroceryRecommender()
    print("✅ Recommender initialized successfully!")
    print(f"📊 Total products: {len(recommender.df)}")
    print(f"🏷️  Categories: {len(recommender.get_categories())}")
    print(f"🧠 Model trained: {recommender.is_trained}")
except Exception as e:
    print(f"❌ Error initializing recommender: {e}")
    recommender = None

# Routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/products')
def get_products():
    """Get products with optional filtering"""
    try:
        if not recommender:
            return jsonify({
                'success': False,
                'error': 'Recommender not initialized',
                'products': []
            }), 500
        
        # Get query parameters
        query = request.args.get('q', '').strip()
        category = request.args.get('category', 'All').strip()
        
        # Get products
        products = recommender.search_products(query=query, category=category, limit=100)
        
        return jsonify({
            'success': True,
            'count': len(products),
            'products': products
        })
        
    except Exception as e:
        print(f"❌ Error in /api/products: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'products': []
        }), 500

@app.route('/api/categories')
def get_categories():
    """Get all product categories"""
    try:
        if not recommender:
            return jsonify({
                'success': False,
                'error': 'Recommender not initialized',
                'categories': []
            }), 500
        
        categories = recommender.get_categories()
        
        return jsonify({
            'success': True,
            'categories': categories
        })
        
    except Exception as e:
        print(f"❌ Error in /api/categories: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'categories': []
        }), 500

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get AI recommendations for selected products"""
    try:
        if not recommender:
            return jsonify({
                'success': False,
                'error': 'Recommender not initialized',
                'recommendations': []
            }), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided',
                'recommendations': []
            }), 400
        
        selected_products = data.get('selected_products', [])
        
        if not selected_products:
            return jsonify({
                'success': False,
                'error': 'No products selected',
                'recommendations': []
            }), 400
        
        # Get recommendations
        recommendations = recommender.recommend(selected_products, n_recommendations=6)
        
        return jsonify({
            'success': True,
            'count': len(recommendations),
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"❌ Error in /api/recommend: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'recommendations': []
        }), 500

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    try:
        if not recommender:
            return jsonify({
                'success': False,
                'error': 'Recommender not initialized',
                'stats': {}
            }), 500
        
        stats = recommender.get_product_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        print(f"❌ Error in /api/stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'stats': {}
        }), 500

@app.route('/api/test')
def test():
    """Test endpoint to verify server is running"""
    return jsonify({
        'success': True,
        'message': 'AI Grocery Recommender API is running!',
        'recommender_initialized': recommender is not None,
        'model_trained': recommender.is_trained if recommender else False
    })

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return app.send_static_file(path)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING FLASK SERVER")
    print("="*60)
    print("🌐 Server URL: http://localhost:5000")
    print("🔗 API Base URL: http://localhost:5000/api")
    print("\n📡 Available Endpoints:")
    print("   GET  /                    - Main page")
    print("   GET  /api/products        - Get products")
    print("   GET  /api/categories      - Get categories")
    print("   POST /api/recommend       - Get recommendations")
    print("   GET  /api/stats           - Get statistics")
    print("   GET  /api/test            - Test endpoint")
    print("\n⚡ Press Ctrl+C to stop the server")
    print("="*60)
    print()
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)