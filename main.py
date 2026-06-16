import numpy as np

# Dictionary mapping string colors to numerical weights
COLOR_MAP = dict(red=1, blue=2, green=3, yellow=4, black=5, white=6, pink=7, purple=8)

class Item:
    """
    Represents a single clothing item with its specific attributes.
    """
    def __init__(self, price: float, is_streetwear: int, is_pants: int, is_accessory: int, color: str):
        self.price = price
        self.is_streetwear = is_streetwear
        self.is_pants = is_pants
        self.is_accessory = is_accessory
        # Convert the string color to its numerical representation
        self.color = COLOR_MAP[color.lower()]

# The main database/catalog of available items
CATALOG = {
    "jogging_nike": Item(price=65.0, is_streetwear=1, is_pants=1, is_accessory=0, color="black"),
    "pantalon_costume": Item(price=120.0, is_streetwear=0, is_pants=1, is_accessory=0, color="blue"),
    "bonnet_carhartt": Item(price=25.0, is_streetwear=1, is_pants=0, is_accessory=1, color="yellow"),
    "t_shirt_basique": Item(price=15.0, is_streetwear=0, is_pants=0, is_accessory=0, color="white"),
    "montre_classique": Item(price=250.0, is_streetwear=0, is_pants=0, is_accessory=1, color="black")
}

class RecommendationEngine:
    """
    A mathematical recommendation engine using Cosine Similarity to find
    and suggest the most similar items based on their numerical features.
    """
    def __init__(self, target_item_name: str):
        """
        Initializes the engine and automatically generates recommendations.
        """
        self.items = list(CATALOG.values())
        self.item_names = list(CATALOG.keys())
        
        # Determine price range for Min-Max scaling
        self.max_price = max([item.price for item in self.items])
        self.min_price = min([item.price for item in self.items])
        
        # Locate the index of the target item
        self.target_index = self.item_names.index(target_item_name.lower())
        
        # Build matrices and generate final recommendations
        raw_matrix = self._build_feature_matrix()
        self.feature_matrix = self._normalize_matrix(raw_matrix)
        self.similarity_matrix = self._compute_similarity()
        self.recommendations = self._get_top_recommendations()

    def _scale_price(self, price: float) -> float:
        """Applies Min-Max scaling to keep the price attribute between 0 and 1."""
        return (price - self.min_price) / (self.max_price - self.min_price)
    
    def _extract_features(self, item: Item) -> list:
        """Converts an Item object into a standardized list of numerical features."""
        return [
            self._scale_price(item.price), 
            item.is_streetwear, 
            item.is_pants, 
            item.is_accessory, 
            item.color
        ]
    
    def _build_feature_matrix(self) -> np.ndarray:
        """
        Iterates over the catalog to build the 2D base matrix of features.
        
        Returns:
            np.ndarray: An unnormalized 2D matrix containing all item attributes.
        """
        matrix_data = []
        for item in self.items:
            features = self._extract_features(item)
            matrix_data.append(features)
            
        return np.array(matrix_data, dtype=float)

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalizes the rows of the matrix to a unit norm (L2 norm) 
        to prepare for Cosine Similarity computation.
        """
        if isinstance(matrix, list):
            matrix = np.array(matrix, dtype=float)
            
        # Calculate the Euclidean norm (L2) for each row
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        
        # Add a tiny epsilon to prevent ZeroDivisionError for items with no features
        norms = np.maximum(norms, 1e-9)
        
        normalized_matrix = matrix / norms
        return normalized_matrix
        
    def _compute_similarity(self) -> np.ndarray:
        """
        Computes the Cosine Similarity for all items simultaneously 
        using matrix multiplication (Dot Product of the matrix and its transpose).
        """
        return self.feature_matrix @ self.feature_matrix.T
    
    def _get_top_recommendations(self) -> list:
        """
        Extracts the top 3 most similar items to the target item using 
        the highly efficient argpartition algorithm (O(N) time complexity).
        
        Returns:
            list: The names of the top 3 recommended items.
        """
        # Isolate the similarity scores for the target item
        scores = self.similarity_matrix[self.target_index]
        
        # We need the Top 4 (The target item itself + 3 recommendations)
        k_closest = min(4, len(scores))

        # 1. Partition the array to push the top K values to the end (unsorted)
        unsorted_top_k_indices = np.argpartition(scores, -k_closest)[-k_closest:]
        
        # 2. Retrieve their actual score values
        top_k_scores = scores[unsorted_top_k_indices]

        # 3. Sort this tiny subset of scores in descending order (highest similarity first)
        sort_order = np.argsort(top_k_scores)[::-1]
        sorted_top_k_indices = unsorted_top_k_indices[sort_order]

        # 4. Remove the target item (which is always at index 0 because score is 1.0)
        top_3_indices = sorted_top_k_indices[1:]

        # 5. Map the mathematical indices back to their string names in O(1)
        recommended_names = []
        for idx in top_3_indices:
            recommended_names.append(self.item_names[idx])
            
        return recommended_names

# ==========================================
# EXECUTION & TESTING
# ==========================================
if __name__ == "__main__":
    # Query the engine
    target = "jogging_nike"
    system = RecommendationEngine(target)
    
    # Display the results
    print(f"Target Item: {target.upper()}")
    print(f"Top 3 Recommendations: {system.recommendations}")