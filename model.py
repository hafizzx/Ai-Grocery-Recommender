import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class GroceryRecommender:
    def __init__(self):
        """Initialize the recommender system with 100 products"""
        print("🎯 Initializing Grocery Recommender...")
        
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Create the dataset with 10 categories × 10 products
        self.create_dataset()
        
        # Prepare and train the model
        self.prepare_data()
        self.train_model()
        
        print(f"✅ Model initialized with {len(self.df)} products across {len(self.get_categories())} categories")
    
    def create_dataset(self):
        """Create a dataset with 100 products (10 categories × 10 products each)"""
        print("📊 Creating product dataset...")
        
        # 10 Categories
        categories = [
            'Fruits', 'Vegetables', 'Dairy', 'Bakery', 'Meat',
            'Beverages', 'Snacks', 'Frozen', 'Personal Care', 'Grains'
        ]
        
        # Product names for each category (10 per category)
        products_by_category = {
            'Fruits': ['Apple', 'Banana', 'Orange', 'Grapes', 'Strawberries', 
                      'Mango', 'Pineapple', 'Watermelon', 'Kiwi', 'Papaya'],
            'Vegetables': ['Carrot', 'Tomato', 'Potato', 'Onion', 'Broccoli',
                          'Spinach', 'Bell Pepper', 'Cucumber', 'Lettuce', 'Cauliflower'],
            'Dairy': ['Milk', 'Eggs', 'Yogurt', 'Cheese', 'Butter',
                     'Cream', 'Cottage Cheese', 'Sour Cream', 'Mozzarella', 'Parmesan'],
            'Bakery': ['Bread', 'Croissant', 'Bagel', 'Muffin', 'Donut',
                      'Cookies', 'Cake', 'Baguette', 'Pita Bread', 'Brown Bread'],
            'Meat': ['Chicken Breast', 'Beef Steak', 'Salmon', 'Pork Chops', 'Turkey',
                    'Bacon', 'Sausages', 'Shrimp', 'Tuna', 'Cod Fillet'],
            'Beverages': ['Coffee', 'Tea', 'Orange Juice', 'Apple Juice', 'Water',
                         'Soda', 'Energy Drink', 'Green Tea', 'Iced Tea', 'Coffee Beans'],
            'Snacks': ['Chips', 'Chocolate', 'Cookies', 'Popcorn', 'Nuts',
                      'Crackers', 'Granola Bar', 'Pretzels', 'Trail Mix', 'Chocolate Bar'],
            'Frozen': ['Frozen Pizza', 'Ice Cream', 'Frozen Vegetables', 'Frozen Berries', 'Frozen Fries',
                      'Frozen Chicken', 'Frozen Fish', 'Frozen Lasagna', 'Frozen Waffles', 'Frozen Meals'],
            'Personal Care': ['Shampoo', 'Soap', 'Toothpaste', 'Deodorant', 'Lotion',
                            'Razor', 'Face Wash', 'Sunscreen', 'Shaving Cream', 'Body Wash'],
            'Grains': ['Rice', 'Pasta', 'Flour', 'Sugar', 'Cereal',
                      'Oats', 'Beans', 'Lentils', 'Quinoa', 'Spices']
        }
        
        data = []
        product_id = 1
        
        # Generate data for each category
        for category in categories:
            products = products_by_category[category]
            
            for i, product_name in enumerate(products):
                # Generate realistic data based on category
                if category == 'Fruits':
                    price = round(np.random.uniform(1.0, 5.0), 2)
                    rating = round(np.random.uniform(4.2, 4.9), 1)
                    popularity = np.random.randint(85, 98)
                elif category == 'Vegetables':
                    price = round(np.random.uniform(0.5, 4.0), 2)
                    rating = round(np.random.uniform(4.0, 4.8), 1)
                    popularity = np.random.randint(80, 95)
                elif category == 'Dairy':
                    price = round(np.random.uniform(2.0, 8.0), 2)
                    rating = round(np.random.uniform(4.3, 4.9), 1)
                    popularity = np.random.randint(85, 97)
                elif category == 'Bakery':
                    price = round(np.random.uniform(1.5, 20.0), 2)
                    rating = round(np.random.uniform(4.1, 4.8), 1)
                    popularity = np.random.randint(75, 95)
                elif category == 'Meat':
                    price = round(np.random.uniform(5.0, 20.0), 2)
                    rating = round(np.random.uniform(4.2, 4.9), 1)
                    popularity = np.random.randint(80, 95)
                elif category == 'Beverages':
                    price = round(np.random.uniform(1.0, 10.0), 2)
                    rating = round(np.random.uniform(4.0, 4.8), 1)
                    popularity = np.random.randint(75, 96)
                elif category == 'Snacks':
                    price = round(np.random.uniform(1.0, 8.0), 2)
                    rating = round(np.random.uniform(4.2, 4.9), 1)
                    popularity = np.random.randint(85, 99)
                elif category == 'Frozen':
                    price = round(np.random.uniform(3.0, 15.0), 2)
                    rating = round(np.random.uniform(3.8, 4.7), 1)
                    popularity = np.random.randint(70, 95)
                elif category == 'Personal Care':
                    price = round(np.random.uniform(2.0, 15.0), 2)
                    rating = round(np.random.uniform(4.0, 4.8), 1)
                    popularity = np.random.randint(75, 93)
                else:  # Grains
                    price = round(np.random.uniform(1.0, 8.0), 2)
                    rating = round(np.random.uniform(4.1, 4.7), 1)
                    popularity = np.random.randint(78, 94)
                
                # Create product entry
                product = {
                    'id': product_id,
                    'name': f"{product_name}",
                    'category': category,
                    'price': price,
                    'rating': rating,
                    'popularity': popularity
                }
                
                data.append(product)
                product_id += 1
        
        # Create DataFrame
        self.df = pd.DataFrame(data)
        print(f"✅ Created dataset with {len(self.df)} products")
    
    def prepare_data(self):
        """Prepare data for KNN algorithm"""
        print("⚙️  Preparing data for AI model...")
        
        # Ensure numeric types
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce').fillna(0)
        self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce').fillna(0)
        self.df['popularity'] = pd.to_numeric(self.df['popularity'], errors='coerce').fillna(0)
        
        # Create enhanced features
        # 1. Category encoding
        category_encoded = pd.get_dummies(self.df['category'], prefix='cat')
        
        # 2. Price tiers (low, medium, high)
        price_q1 = self.df['price'].quantile(0.33)
        price_q2 = self.df['price'].quantile(0.67)
        self.df['price_tier'] = self.df['price'].apply(
            lambda x: 0 if x <= price_q1 else (1 if x <= price_q2 else 2)
        )
        
        # 3. Rating tiers
        rating_q1 = self.df['rating'].quantile(0.33)
        rating_q2 = self.df['rating'].quantile(0.67)
        self.df['rating_tier'] = self.df['rating'].apply(
            lambda x: 0 if x <= rating_q1 else (1 if x <= rating_q2 else 2)
        )
        
        # 4. Popularity tiers
        pop_q1 = self.df['popularity'].quantile(0.33)
        pop_q2 = self.df['popularity'].quantile(0.67)
        self.df['popularity_tier'] = self.df['popularity'].apply(
            lambda x: 0 if x <= pop_q1 else (1 if x <= pop_q2 else 2)
        )
        
        # Combine all features
        numeric_features = self.df[['price', 'rating', 'popularity', 
                                   'price_tier', 'rating_tier', 'popularity_tier']].copy()
        
        self.features = pd.concat([numeric_features, category_encoded], axis=1)
        
        # Scale features
        self.scaled_features = self.scaler.fit_transform(self.features)
        
        print(f"✅ Prepared {self.features.shape[1]} features for AI model")
    
    def train_model(self, n_neighbors=6):
        """Train the KNN model"""
        try:
            # Adjust neighbors based on dataset size
            n_neighbors = min(n_neighbors + 1, max(3, len(self.df)))
            
            self.model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric='euclidean',  # Using Euclidean for better results
                algorithm='auto'
            )
            self.model.fit(self.scaled_features)
            self.is_trained = True
            print(f"✅ KNN model trained with {n_neighbors-1} neighbors")
        except Exception as e:
            print(f"❌ Error training model: {e}")
            self.is_trained = False
    
    def find_product_by_name(self, product_name):
        """Find product index by name (exact or partial match)"""
        if not product_name:
            return None
        
        product_name_lower = str(product_name).lower().strip()
        
        # Exact match
        exact_match = self.df[self.df['name'].str.lower() == product_name_lower]
        if not exact_match.empty:
            return exact_match.index[0]
        
        # Partial match
        partial_match = self.df[self.df['name'].str.lower().str.contains(product_name_lower)]
        if not partial_match.empty:
            return partial_match.index[0]
        
        return None
    
    def recommend(self, product_names, n_recommendations=6):
        """Get AI recommendations based on selected products"""
        print(f"🎯 Getting recommendations for: {product_names}")
        
        if not self.is_trained or not product_names:
            return []
        
        all_recommendations = []
        
        for product_name in product_names:
            product_idx = self.find_product_by_name(product_name)
            
            if product_idx is not None:
                # Get neighbors
                distances, indices = self.model.kneighbors(
                    [self.scaled_features[product_idx]], 
                    n_neighbors=min(15, len(self.df))
                )
                
                # Process neighbors
                for i, neighbor_idx in enumerate(indices[0]):
                    if neighbor_idx == product_idx:  # Skip itself
                        continue
                    
                    neighbor_product = self.df.iloc[neighbor_idx]
                    
                    # Skip if already in selected products
                    if neighbor_product['name'] in product_names:
                        continue
                    
                    # Calculate similarity score (0-100)
                    similarity = max(0, 100 - distances[0][i] * 20)
                    
                    # Add bonus for same category (encourage category coherence)
                    selected_product = self.df.iloc[product_idx]
                    if neighbor_product['category'] == selected_product['category']:
                        similarity += 5
                    
                    # Add bonus for price similarity
                    price_diff = abs(neighbor_product['price'] - selected_product['price'])
                    if price_diff <= 2:
                        similarity += 3
                    
                    recommendation = {
                        'id': int(neighbor_product['id']),
                        'name': neighbor_product['name'],
                        'category': neighbor_product['category'],
                        'price': float(neighbor_product['price']),
                        'rating': float(neighbor_product['rating']),
                        'popularity': int(neighbor_product['popularity']),
                        'score': float(min(100, round(similarity, 1)))  # Cap at 100
                    }
                    
                    all_recommendations.append(recommendation)
        
        # Remove duplicates and sort by score
        unique_recommendations = []
        seen_ids = set()
        
        for rec in sorted(all_recommendations, key=lambda x: x['score'], reverse=True):
            if rec['id'] not in seen_ids:
                seen_ids.add(rec['id'])
                unique_recommendations.append(rec)
                
                if len(unique_recommendations) >= n_recommendations:
                    break
        
        print(f"✅ Generated {len(unique_recommendations)} recommendations")
        return unique_recommendations
    
    def search_products(self, query="", category="All", limit=100):
        """Search products with filters"""
        results = self.df.copy()
        
        # Apply search query
        if query and query.strip():
            query_lower = query.lower().strip()
            mask = results['name'].str.lower().str.contains(query_lower, na=False)
            results = results[mask]
        
        # Apply category filter
        if category and category != "All":
            results = results[results['category'] == category]
        
        # Sort by popularity, rating, and price
        results = results.sort_values(
            ['popularity', 'rating', 'price'], 
            ascending=[False, False, True]
        )
        
        return results.head(limit).to_dict('records')
    
    def get_categories(self):
        """Get all unique categories"""
        if self.df is not None and 'category' in self.df.columns:
            categories = self.df['category'].dropna().unique().tolist()
            return sorted(categories)
        return []
    
    def get_product_stats(self):
        """Get product statistics"""
        try:
            stats = {
                'total_products': len(self.df),
                'total_categories': len(self.get_categories()),
                'avg_price': round(self.df['price'].mean(), 2),
                'avg_rating': round(self.df['rating'].mean(), 2),
                'avg_popularity': round(self.df['popularity'].mean(), 2)
            }
            return stats
        except:
            return {
                'total_products': len(self.df) if self.df is not None else 0,
                'total_categories': len(self.get_categories()),
                'avg_price': 0,
                'avg_rating': 0,
                'avg_popularity': 0
            }