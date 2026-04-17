"""Real-time content filtering for blocking harmful content before display."""

import logging
from typing import Dict, List, Any
import tensorflow as tf
from pathlib import Path

logger = logging.getLogger(__name__)


class ContentFilter:
    """Multimodal content filter for real-time harmful content detection."""

    def __init__(self, text_model_path: str = None, link_model_path: str = None, image_model_path: str = None):
        """Initialize content filter with trained models.
        
        Args:
            text_model_path: Path to trained text classifier
            link_model_path: Path to trained link analyzer
            image_model_path: Path to trained image filter
        """
        self.text_model = None
        self.link_model = None
        self.image_model = None
        self.confidence_threshold = 0.7
        
        # Load models if paths provided
        if text_model_path:
            self.load_text_model(text_model_path)
        if link_model_path:
            self.load_link_model(link_model_path)
        if image_model_path:
            self.load_image_model(image_model_path)

    def load_text_model(self, model_path: str):
        """Load trained text classification model.
        
        Args:
            model_path: Path to model
        """
        try:
            self.text_model = tf.keras.models.load_model(model_path)
            logger.info(f"Text model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading text model: {e}")

    def load_link_model(self, model_path: str):
        """Load trained link analyzer model.
        
        Args:
            model_path: Path to model
        """
        try:
            self.link_model = tf.keras.models.load_model(model_path)
            logger.info(f"Link model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading link model: {e}")

    def load_image_model(self, model_path: str):
        """Load trained image filter model.
        
        Args:
            model_path: Path to model
        """
        try:
            self.image_model = tf.keras.models.load_model(model_path)
            logger.info(f"Image model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading image model: {e}")

    def filter_text(self, text: str) -> Dict[str, Any]:
        """Filter text content.
        
        Args:
            text: Text content to filter
            
        Returns:
            Dictionary with filtering result
        """
        if not self.text_model:
            return {"error": "Text model not loaded", "safe": True}
        
        try:
            # Preprocess and predict
            # Note: In production, use the same preprocessing as training
            prediction = self.text_model.predict([text], verbose=0)[0][0]
            
            is_safe = prediction < self.confidence_threshold
            confidence = max(prediction, 1 - prediction)
            
            result = {
                "safe": is_safe,
                "confidence": float(confidence),
                "risk_score": float(prediction),
                "content_type": "text"
            }
            
            if not is_safe:
                logger.warning(f"Harmful text detected. Risk score: {prediction:.4f}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error filtering text: {e}")
            return {"error": str(e), "safe": True}

    def filter_link(self, url: str, url_analysis: Dict = None) -> Dict[str, Any]:
        """Filter link/URL content.
        
        Args:
            url: URL to filter
            url_analysis: Pre-computed URL analysis (optional)
            
        Returns:
            Dictionary with filtering result
        """
        if not self.link_model:
            return {"error": "Link model not loaded", "safe": True}
        
        try:
            # If analysis provided, use it; otherwise use basic checks
            if url_analysis:
                features = self._extract_link_features(url_analysis)
            else:
                features = self._extract_basic_link_features(url)
            
            prediction = self.link_model.predict([features], verbose=0)[0][0]
            
            is_safe = prediction < self.confidence_threshold
            confidence = max(prediction, 1 - prediction)
            
            result = {
                "safe": is_safe,
                "confidence": float(confidence),
                "risk_score": float(prediction),
                "risk_level": "high" if prediction > 0.8 else "medium" if prediction > 0.5 else "low",
                "content_type": "link"
            }
            
            if not is_safe:
                logger.warning(f"Harmful link detected: {url}. Risk score: {prediction:.4f}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error filtering link: {e}")
            return {"error": str(e), "safe": True}

    def filter_image(self, image_path: str) -> Dict[str, Any]:
        """Filter image content.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with filtering result
        """
        if not self.image_model:
            return {"error": "Image model not loaded", "safe": True}
        
        try:
            # Load and preprocess image
            import tensorflow as tf
            from PIL import Image
            
            img = Image.open(image_path).resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array = img_array / 255.0  # Normalize
            
            prediction = self.image_model.predict(img_array, verbose=0)[0][0]
            
            is_safe = prediction < self.confidence_threshold
            confidence = max(prediction, 1 - prediction)
            
            result = {
                "safe": is_safe,
                "confidence": float(confidence),
                "risk_score": float(prediction),
                "content_type": "image"
            }
            
            if not is_safe:
                logger.warning(f"Harmful image detected. Risk score: {prediction:.4f}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error filtering image: {e}")
            return {"error": str(e), "safe": True}

    def filter_multimodal(self, content: Dict) -> Dict[str, Any]:
        """Filter multimodal content (text + links + images).
        
        Args:
            content: Dictionary with 'text', 'links', 'images' keys
            
        Returns:
            Aggregated filtering results
        """
        results = {
            "text_results": [],
            "link_results": [],
            "image_results": [],
            "overall_safe": True,
            "blocking_reasons": []
        }
        
        # Filter text content
        if 'text' in content:
            for text in content.get('text', []):
                text_result = self.filter_text(text)
                results["text_results"].append(text_result)
                if not text_result.get('safe', True):
                    results["overall_safe"] = False
                    results["blocking_reasons"].append("harmful_text")
        
        # Filter links
        if 'links' in content:
            for link in content.get('links', []):
                link_result = self.filter_link(link)
                results["link_results"].append(link_result)
                if not link_result.get('safe', True):
                    results["overall_safe"] = False
                    results["blocking_reasons"].append(f"harmful_link_{link}")
        
        # Filter images
        if 'images' in content:
            for image in content.get('images', []):
                image_result = self.filter_image(image)
                results["image_results"].append(image_result)
                if not image_result.get('safe', True):
                    results["overall_safe"] = False
                    results["blocking_reasons"].append("harmful_image")
        
        return results

    def _extract_link_features(self, url_analysis: Dict) -> List[float]:
        """Extract features from URL analysis."""
        features = [
            len(url_analysis.get('url', '')),
            len(url_analysis.get('domain', '')),
            len(url_analysis.get('path', '')),
            len(url_analysis.get('query_params', {})),
            1 if url_analysis.get('has_suspicious_keywords') else 0,
            len(url_analysis.get('suspicious_keywords', [])),
            1 if url_analysis.get('is_suspiciously_long') else 0,
            1 if url_analysis.get('has_excess_special_chars') else 0,
            1 if url_analysis.get('uses_ip_address') else 0,
            url_analysis.get('risk_score', 0),
        ]
        return features

    def _extract_basic_link_features(self, url: str) -> List[float]:
        """Extract basic features from URL string."""
        features = [
            len(url),
            url.count('/'),
            url.count('?'),
            url.count('@'),
            url.count('-'),
            url.count('_'),
            1 if any(kw in url.lower() for kw in ['adult', 'xxx', 'porn']) else 0,
            1 if len(url) > 200 else 0,
            1 if any(c in url for c in ['!', '#', '$', '%']) else 0,
            0.0  # Placeholder for model consistency
        ]
        return features

    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for blocking.
        
        Args:
            threshold: Value between 0 and 1
        """
        if 0 <= threshold <= 1:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold set to {threshold}")
        else:
            logger.error("Threshold must be between 0 and 1")
