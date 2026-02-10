import json
import logging
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("phishing_detector.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Phishing URL Detector", page_icon="🔒", layout="centered"
)

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        padding: 20px 0;
    }
    .url-input {
        font-size: 16px;
        padding: 10px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
    }
    .phishing {
        background-color: #ffcccc;
        color: #cc0000;
        border: 2px solid #cc0000;
    }
    .legitimate {
        background-color: #ccffcc;
        color: #006600;
        border: 2px solid #006600;
    }
    .confidence-bar {
        margin: 10px 0;
    }
    .footer {
        text-align: center;
        color: #666;
        padding: 20px 0;
        font-size: 14px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_components():
    """Load model, scaler, and metadata"""
    logger.info("=" * 80)
    logger.info("LOADING MODEL COMPONENTS")
    logger.info("=" * 80)
    try:
        logger.info("Loading phishing_model.pkl...")
        model = joblib.load("phishing_model.pkl")
        logger.info(f"✓ Model loaded: {type(model).__name__}")

        logger.info("Loading scaler.pkl...")
        scaler = joblib.load("scaler.pkl")
        logger.info(f"✓ Scaler loaded: {type(scaler).__name__}")

        logger.info("Loading model_metadata.json...")
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)
        logger.info(
            f"✓ Metadata loaded: {len(metadata.get('feature_names', []))} features"
        )
        logger.info(f"  Model name: {metadata.get('model_name', 'Unknown')}")
        logger.info(f"  Test recall: {metadata.get('test_recall', 0):.4f}")
        logger.info(f"  Test F1: {metadata.get('test_f1', 0):.4f}")
        logger.info("=" * 80)

        return model, scaler, metadata
    except FileNotFoundError as e:
        st.error(f"""
        ⚠️ Model files not found! 
        
        Please run the model training notebook first and execute the last cell to export:
        - phishing_model.pkl
        - scaler.pkl
        - model_metadata.json
        
        Error: {str(e)}
        """)
        st.stop()


@st.cache_data
def load_training_data():
    """Load training dataset for quick entry features"""
    logger.info("Attempting to load training CSV dataset...")
    try:
        df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
        logger.info(
            f"✓ CSV loaded successfully: {len(df)} rows, {len(df.columns)} columns"
        )
        logger.info(
            f"  CSV columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}"
        )
        if "label" in df.columns:
            logger.info(f"  Phishing (0): {(df['label'] == 0).sum()}")
            logger.info(f"  Legitimate (1): {(df['label'] == 1).sum()}")
        return df
    except FileNotFoundError:
        logger.warning("⚠ CSV file 'PhiUSIIL_Phishing_URL_Dataset.csv' not found")
        return None
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        return None


def get_random_csv_entry(df, feature_names):
    """Get a random entry from the CSV dataset"""
    logger.info("📊 Loading random CSV entry...")
    if df is None:
        logger.warning("  DataFrame is None, cannot load entry")
        return None, None

    # Select a random row
    random_idx = random.randint(0, len(df) - 1)
    row = df.iloc[random_idx]
    logger.info(f"  Selected row index: {random_idx}")

    # Extract URL if available
    url = row.get("URL", "") if "URL" in df.columns else ""
    logger.info(f"  URL: {url}")

    # Extract manual features that are in the dataset
    manual_features = {}
    for feature in feature_names:
        if feature in df.columns:
            value = row[feature]
            # Convert to appropriate type
            if pd.notna(value):
                manual_features[feature] = (
                    int(value) if isinstance(value, (int, np.integer)) else float(value)
                )

    logger.info(f"  Extracted {len(manual_features)} features from CSV")
    if "label" in row:
        logger.info(
            f"  Actual label in CSV: {row['label']} ({'Phishing' if row['label'] == 0 else 'Legitimate'})"
        )

    return url, manual_features


def get_random_values_in_range(df, feature_names):
    """Generate random values within the ranges of training data"""
    if df is None:
        return {}

    manual_features = {}

    # Features that can be manually input
    input_features = [
        "LineOfCode",
        "LargestLineLength",
        "NoOfImage",
        "NoOfCSS",
        "NoOfJS",
        "HasTitle",
        "HasFavicon",
        "HasDescription",
        "HasSocialNet",
        "HasCopyrightInfo",
        "IsResponsive",
        "Robots",
        "NoOfPopup",
        "NoOfiFrame",
        "HasSubmitButton",
        "HasHiddenFields",
        "HasPasswordField",
        "HasExternalFormSubmit",
        "NoOfSelfRef",
        "NoOfExternalRef",
        "NoOfEmptyRef",
        "NoOfURLRedirect",
    ]

    for feature in input_features:
        if feature in df.columns:
            col_data = df[feature].dropna()
            if len(col_data) > 0:
                min_val = col_data.min()
                max_val = col_data.max()

                # For binary features (0 or 1)
                if feature.startswith("Has") or feature in ["IsResponsive", "Robots"]:
                    manual_features[feature] = random.choice([0, 1])
                else:
                    # For numeric features, generate random value in range
                    manual_features[feature] = random.randint(
                        int(min_val), int(max_val)
                    )

    return manual_features


def extract_features_from_url(url_string, manual_features=None):
    """
    Extract features from URL matching the training dataset structure.

    Extracts 46 features as defined in model_metadata.json:
    - URL structure features (domain, TLD, subdomains, etc.)
    - Character analysis (letters, digits, special chars)
    - Domain and security features (HTTPS, IP, obfuscation)
    - Content features (approximated with heuristics or manual input)

    Args:
        url_string: The URL to analyze
        manual_features: Optional dict of manually-provided feature values
                        If this contains most/all of the 46 features (from CSV),
                        it will be used directly without URL extraction.

    Note: Some features require actual webpage content (HTML parsing, external requests).
    Those are approximated with heuristics or can be manually provided.
    """
    import re
    from urllib.parse import urlparse

    logger.info("=" * 80)
    logger.info("FEATURE EXTRACTION")
    logger.info("=" * 80)
    logger.info(f"URL: {url_string}")
    logger.info(
        f"Manual features provided: {len(manual_features) if manual_features else 0}"
    )

    if manual_features is None:
        manual_features = {}

    # Get feature names from session state
    feature_names = st.session_state.metadata["feature_names"]
    logger.info(f"Expected feature count: {len(feature_names)}")

    # CRITICAL FIX: If manual_features contains most features (from CSV quick entry),
    # use them directly without mixing with URL extraction
    if len(manual_features) >= 40:  # Most features are provided (likely from CSV)
        logger.info("Using CSV-provided features directly (40+ features available)")
        feature_vector = []
        for fname in feature_names:
            if fname in manual_features:
                feature_vector.append(manual_features[fname])
            else:
                # Use 0 as default for missing features
                feature_vector.append(0)

        result_df = pd.DataFrame([feature_vector], columns=feature_names)
        logger.info("Feature extraction complete (CSV mode)")
        logger.info(f"Feature vector shape: {result_df.shape}")
        logger.info("First 10 features:")
        for i, (name, val) in enumerate(zip(feature_names[:10], feature_vector[:10])):
            logger.info(f"  {name}: {val}")
        logger.info("=" * 80)

        # Return DataFrame directly with CSV features
        return result_df

    try:
        # Parse URL
        parsed = urlparse(url_string)
        domain = parsed.netloc

        # Initialize feature dictionary
        features = {}

        # ===== BASIC URL FEATURES =====
        features["DomainLength"] = len(domain)

        # Check if domain is IP address
        ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
        features["IsDomainIP"] = 1 if re.match(ip_pattern, domain) else 0

        # Extract TLD
        tld = domain.split(".")[-1] if "." in domain else ""
        features["TLDLength"] = len(tld)

        # URL Similarity Index (simplified - would need comparison with known sites)
        features["URLSimilarityIndex"] = 0.5  # Neutral value

        # Character Continuation Rate (repeated characters)
        max_continuation = max(
            [len(list(g)) for k, g in __import__("itertools").groupby(url_string)],
            default=1,
        )
        features["CharContinuationRate"] = (
            max_continuation / len(url_string) if len(url_string) > 0 else 0
        )

        # TLD Legitimacy Probability (heuristic based on common TLDs)
        common_tlds = ["com", "org", "net", "edu", "gov", "co", "uk", "io"]
        features["TLDLegitimateProb"] = 0.8 if tld.lower() in common_tlds else 0.3

        # URL Character Probability (simplified heuristic)
        features["URLCharProb"] = 0.5  # Neutral value

        # Number of subdomains
        features["NoOfSubDomain"] = (
            len(domain.split(".")) - 2 if len(domain.split(".")) > 2 else 0
        )

        # ===== OBFUSCATION FEATURES =====
        # Check for obfuscation (URL encoding, excessive special chars)
        obfuscation_chars = ["%", "@", "\\\\"]
        has_obfuscation = any(char in url_string for char in obfuscation_chars)
        features["HasObfuscation"] = 1 if has_obfuscation else 0
        features["NoOfObfuscatedChar"] = sum(
            url_string.count(char) for char in obfuscation_chars
        )
        features["ObfuscationRatio"] = (
            features["NoOfObfuscatedChar"] / len(url_string)
            if len(url_string) > 0
            else 0
        )

        # ===== CHARACTER ANALYSIS =====
        features["NoOfLettersInURL"] = sum(c.isalpha() for c in url_string)
        features["LetterRatioInURL"] = (
            features["NoOfLettersInURL"] / len(url_string) if len(url_string) > 0 else 0
        )

        features["DegitRatioInURL"] = (
            sum(c.isdigit() for c in url_string) / len(url_string)
            if len(url_string) > 0
            else 0
        )

        features["NoOfQMarkInURL"] = url_string.count("?")
        features["NoOfAmpersandInURL"] = url_string.count("&")

        # Other special characters
        special_chars = set("!#$%^*()_+-[]{}|;:,.<>/?~`")
        features["NoOfOtherSpecialCharsInURL"] = sum(
            url_string.count(c) for c in special_chars
        )
        features["SpacialCharRatioInURL"] = (
            (
                features["NoOfOtherSpecialCharsInURL"]
                + url_string.count("=")  # NoOfEqualsInURL (not in final features)
                + features["NoOfQMarkInURL"]
                + features["NoOfAmpersandInURL"]
            )
            / len(url_string)
            if len(url_string) > 0
            else 0
        )

        # ===== SECURITY FEATURES =====
        features["IsHTTPS"] = 1 if parsed.scheme == "https" else 0

        # ===== CONTENT FEATURES (HEURISTIC-BASED) =====
        # These features normally require fetching webpage content
        # We use URL characteristics to make educated guesses about page properties

        # Calculate heuristic scores based on URL characteristics
        url_length = len(url_string)
        domain_parts = domain.split(".")
        has_suspicious_keywords = any(
            kw in url_string.lower()
            for kw in [
                "verify",
                "secure",
                "account",
                "update",
                "confirm",
                "login",
                "signin",
                "banking",
                "suspend",
                "unusual",
                "click",
                "urgent",
            ]
        )

        # Known legitimate domains (very simplified whitelist)
        known_legit = any(
            legit_domain in domain.lower()
            for legit_domain in [
                "google.com",
                "github.com",
                "wikipedia.org",
                "microsoft.com",
                "amazon.com",
                "facebook.com",
                "twitter.com",
                "linkedin.com",
                "apple.com",
                "netflix.com",
                "youtube.com",
            ]
        )

        # Content features based on URL analysis
        # Suspicious URLs tend to have more complex pages (higher values)
        # Legitimate well-known sites tend to have lower values

        if known_legit:
            # Known legitimate sites: use legitimate-typical values
            features["LineOfCode"] = manual_features.get(
                "LineOfCode", 50 + (url_length % 50)
            )
            features["LargestLineLength"] = manual_features.get(
                "LargestLineLength", 1000 + (url_length * 10)
            )
            features["HasTitle"] = manual_features.get("HasTitle", 1)
            features["DomainTitleMatchScore"] = 0.8
            features["HasFavicon"] = manual_features.get("HasFavicon", 0)
            features["Robots"] = manual_features.get("Robots", 0)
            features["IsResponsive"] = manual_features.get("IsResponsive", 0)
            features["HasDescription"] = manual_features.get("HasDescription", 0)
            features["HasSocialNet"] = manual_features.get("HasSocialNet", 0)
            features["HasCopyrightInfo"] = manual_features.get("HasCopyrightInfo", 0)
            features["NoOfImage"] = manual_features.get("NoOfImage", url_length % 10)
            features["NoOfCSS"] = manual_features.get("NoOfCSS", 0)
            features["NoOfJS"] = manual_features.get("NoOfJS", 1)
            features["NoOfSelfRef"] = manual_features.get("NoOfSelfRef", 0)
            features["NoOfExternalRef"] = manual_features.get("NoOfExternalRef", 1)
        else:
            # Unknown or suspicious: use phishing-typical values
            # These increase with URL complexity (length, special chars, subdomains)
            complexity_score = (
                url_length / 50
                + len(domain_parts) * 10
                + features["NoOfObfuscatedChar"] * 20
                + (1 if has_suspicious_keywords else 0) * 50
            )

            features["LineOfCode"] = manual_features.get(
                "LineOfCode", int(1000 + complexity_score * 20)
            )
            features["LargestLineLength"] = manual_features.get(
                "LargestLineLength", int(5000 + complexity_score * 100)
            )
            features["HasTitle"] = manual_features.get("HasTitle", 1)
            features["DomainTitleMatchScore"] = 0.3 if has_suspicious_keywords else 0.5
            features["HasFavicon"] = manual_features.get("HasFavicon", 1)
            features["Robots"] = manual_features.get("Robots", 0)
            features["IsResponsive"] = manual_features.get("IsResponsive", 1)
            features["HasDescription"] = manual_features.get("HasDescription", 1)
            features["HasSocialNet"] = manual_features.get("HasSocialNet", 1)
            features["HasCopyrightInfo"] = manual_features.get("HasCopyrightInfo", 1)
            features["NoOfImage"] = manual_features.get(
                "NoOfImage", int(30 + complexity_score / 5)
            )
            features["NoOfCSS"] = manual_features.get(
                "NoOfCSS", int(8 + complexity_score / 10)
            )
            features["NoOfJS"] = manual_features.get(
                "NoOfJS", int(12 + complexity_score / 8)
            )
            features["NoOfSelfRef"] = manual_features.get(
                "NoOfSelfRef", int(80 + complexity_score * 2)
            )
            features["NoOfExternalRef"] = manual_features.get(
                "NoOfExternalRef", int(60 + complexity_score * 1.5)
            )

        # Common features regardless of domain type
        features["NoOfURLRedirect"] = manual_features.get("NoOfURLRedirect", 0)
        features["NoOfSelfRedirect"] = 0
        features["NoOfPopup"] = manual_features.get("NoOfPopup", 0)
        features["NoOfiFrame"] = manual_features.get(
            "NoOfiFrame", 2 if not known_legit else 0
        )
        features["HasExternalFormSubmit"] = manual_features.get(
            "HasExternalFormSubmit", 0
        )
        features["HasSubmitButton"] = manual_features.get(
            "HasSubmitButton",
            1 if "login" in url_string.lower() or "signin" in url_string.lower() else 0,
        )
        features["HasHiddenFields"] = manual_features.get("HasHiddenFields", 0)
        features["HasPasswordField"] = manual_features.get(
            "HasPasswordField",
            1 if "login" in url_string.lower() or "signin" in url_string.lower() else 0,
        )
        features["NoOfEmptyRef"] = manual_features.get(
            "NoOfEmptyRef", 3 if not known_legit else 0
        )

        # Keyword detection
        features["Bank"] = (
            1
            if any(
                word in url_string.lower() for word in ["bank", "banking", "account"]
            )
            else 0
        )
        features["Pay"] = (
            1
            if any(
                word in url_string.lower() for word in ["pay", "payment", "checkout"]
            )
            else 0
        )
        features["Crypto"] = (
            1
            if any(
                word in url_string.lower() for word in ["crypto", "bitcoin", "wallet"]
            )
            else 0
        )

        # Convert to DataFrame with correct column order
        feature_names = st.session_state.metadata["feature_names"]

        # Create feature vector in correct order
        feature_vector = []
        for fname in feature_names:
            if fname in features:
                feature_vector.append(features[fname])
            else:
                # If feature not extracted, use 0 as default
                feature_vector.append(0)

        # Convert to DataFrame with proper column names to avoid sklearn warnings
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)

        logger.info("Feature extraction complete (URL mode)")
        logger.info(f"Feature vector shape: {feature_df.shape}")
        logger.info("Sample of extracted features (first 15):")
        for i, (name, val) in enumerate(zip(feature_names[:15], feature_vector[:15])):
            logger.info(f"  {name}: {val}")
        logger.info("=" * 80)

        return feature_df

    except Exception as e:
        logger.error(f"❌ Error extracting features: {str(e)}")
        logger.exception("Full traceback:")
        st.error(f"Error extracting features: {str(e)}")
        # Return zeros DataFrame if extraction fails
        feature_names = st.session_state.metadata["feature_names"]
        logger.warning("Returning zero-filled feature vector due to error")
        logger.info("=" * 80)
        return pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)


def predict_phishing(url, model, scaler, metadata, manual_features=None):
    """Make prediction for a given URL.

    Args:
        url: URL string to analyze
        model: Trained ML model
        scaler: Feature scaler (unused for Random Forest, kept for compatibility)
        metadata: Model metadata with feature names (unused but kept for compatibility)
        manual_features: Optional dict of manually provided feature values
    """
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"URL: {url}")

    try:
        # Extract features from URL (with manual overrides if provided)
        features = extract_features_from_url(url, manual_features)

        logger.info("\nPassing features to model...")
        logger.info(f"Feature matrix shape: {features.shape}")
        logger.info(f"Feature data types: {features.dtypes.value_counts().to_dict()}")

        # Random Forest doesn't need scaling - use features directly
        # Make prediction
        logger.info("Making prediction...")
        prediction = model.predict(features)[0]
        logger.info(
            f"✓ Prediction: {prediction} ({'Phishing' if prediction == 0 else 'Legitimate'})"
        )

        # Get probability scores if available
        # IMPORTANT: Label encoding is 0=Phishing, 1=Legitimate
        # So probabilities[0] is phishing probability, probabilities[1] is legitimate
        if hasattr(model, "predict_proba"):
            logger.info("Calculating probability scores...")
            probabilities = model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))
            phishing_prob = float(probabilities[0])  # Probability of phishing (class 0)
            legitimate_prob = float(
                probabilities[1]
            )  # Probability of legitimate (class 1)
            logger.info(
                f"✓ Phishing probability: {phishing_prob:.4f} ({phishing_prob * 100:.2f}%)"
            )
            logger.info(
                f"✓ Legitimate probability: {legitimate_prob:.4f} ({legitimate_prob * 100:.2f}%)"
            )
            logger.info(
                f"✓ Model confidence: {confidence:.4f} ({confidence * 100:.2f}%)"
            )
        else:
            logger.info("Model does not support probability predictions")
            confidence = 1.0
            # prediction==0 means phishing, prediction==1 means legitimate
            phishing_prob = 1.0 if prediction == 0 else 0.0
            legitimate_prob = 1.0 if prediction == 1 else 0.0

        result = {
            "prediction": int(prediction),
            "confidence": confidence,
            "phishing_probability": phishing_prob,
            "legitimate_probability": legitimate_prob,
        }

        logger.info("\n" + "=" * 80)
        logger.info("PREDICTION RESULT SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"Final Classification: {'⚠️  PHISHING' if prediction == 0 else '✅ LEGITIMATE'}"
        )
        logger.info(f"Confidence: {confidence * 100:.2f}%")
        logger.info(f"Phishing Risk: {phishing_prob * 100:.2f}%")
        logger.info("=" * 80 + "\n")

        return result

    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        logger.exception("Full traceback:")
        logger.info("=" * 80 + "\n")
        st.error(f"Error during prediction: {str(e)}")
        return None


# Main app
def main():
    # Load model components
    if "model" not in st.session_state:
        st.session_state.model, st.session_state.scaler, st.session_state.metadata = (
            load_model_components()
        )

    # Header
    st.markdown(
        '<h1 class="main-header">🔒 Phishing URL Detector</h1>', unsafe_allow_html=True
    )
    st.markdown(
        """
    <p style="text-align: center; color: #666; font-size: 16px;">
    Protect yourself from phishing attacks. Enter a URL below to check if it's safe or suspicious.
    </p>
    """,
        unsafe_allow_html=True,
    )

    # Display model info in sidebar
    st.sidebar.header("ℹ️ Model Information")
    st.sidebar.markdown(f"""
    **Model Type:** {st.session_state.metadata["model_name"]}
    
    **Performance Metrics:**
    - Recall: {st.session_state.metadata["test_recall"]:.2%}
    - F1-Score: {st.session_state.metadata["test_f1"]:.4f}
    - ROC-AUC: {st.session_state.metadata["test_roc_auc"]:.4f}
    
    **Features Used:** {len(st.session_state.metadata["feature_names"])}
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🛡️ What is Phishing?
    Phishing is a cybercrime where attackers impersonate legitimate organizations 
    to steal sensitive information like passwords, credit cards, or personal data.
    
    ### 🚨 Warning Signs:
    - Suspicious URLs with misspellings
    - Unusual domain names
    - Requests for personal information
    - Too-good-to-be-true offers
    """)

    # Main input area
    st.markdown("---")

    # URL input
    # Check if quick entry was triggered
    default_url = ""
    if "quick_entry_url" in st.session_state:
        default_url = st.session_state.quick_entry_url

    url_input = st.text_input(
        "🔗 Enter URL to check:",
        value=default_url,
        placeholder="https://example.com",
        help="Enter the complete URL including http:// or https://",
    )

    # Optional: Manual feature input for features that cannot be extracted from URL
    with st.expander("⚙️ Advanced: Manual Feature Input (Optional)"):
        st.markdown("""
        Some features require analyzing the actual webpage content. 
        If you have access to the webpage, you can manually input these features for more accurate predictions.
        """)

        # Quick entry buttons
        st.markdown("**Quick Entry Options:**")
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button(
                "📊 Load Random CSV Entry",
                help="Fill form with a random entry from training data",
            ):
                training_df = load_training_data()
                if training_df is not None:
                    url, features = get_random_csv_entry(
                        training_df, st.session_state.metadata["feature_names"]
                    )
                    if features:
                        st.session_state.quick_entry_url = url
                        st.session_state.quick_entry_features = features
                        st.rerun()
                else:
                    st.warning(
                        "⚠️ CSV file not found. Please ensure PhiUSIIL_Phishing_URL_Dataset.csv is in the directory."
                    )

        with col_btn2:
            if st.button(
                "🎲 Generate Random Values",
                help="Generate random values within training data ranges",
            ):
                training_df = load_training_data()
                if training_df is not None:
                    features = get_random_values_in_range(
                        training_df, st.session_state.metadata["feature_names"]
                    )
                    if features:
                        st.session_state.quick_entry_url = "https://example-random.com"
                        st.session_state.quick_entry_features = features
                        st.rerun()
                else:
                    st.warning(
                        "⚠️ CSV file not found. Please ensure PhiUSIIL_Phishing_URL_Dataset.csv is in the directory."
                    )

        st.markdown("---")

        col_a, col_b, col_c = st.columns(3)

        # Get quick entry features if available
        quick_features = st.session_state.get("quick_entry_features", {})

        with col_a:
            st.markdown("**Page Content**")
            manual_line_of_code = st.number_input(
                "Lines of Code",
                min_value=0,
                value=quick_features.get("LineOfCode", None),
                help="Number of lines in HTML source",
            )
            manual_largest_line = st.number_input(
                "Largest Line Length",
                min_value=0,
                value=quick_features.get("LargestLineLength", None),
                help="Length of longest line in HTML",
            )
            manual_no_of_image = st.number_input(
                "Number of Images",
                min_value=0,
                value=quick_features.get("NoOfImage", None),
                help="Total <img> tags",
            )
            manual_no_of_css = st.number_input(
                "Number of CSS Files",
                min_value=0,
                value=quick_features.get("NoOfCSS", None),
                help="External CSS files linked",
            )
            manual_no_of_js = st.number_input(
                "Number of JS Files",
                min_value=0,
                value=quick_features.get("NoOfJS", None),
                help="External JavaScript files",
            )

        with col_b:
            st.markdown("**Page Elements**")
            manual_has_title = st.checkbox(
                "Has Title Tag",
                value=bool(quick_features.get("HasTitle", 0))
                if "HasTitle" in quick_features
                else None,
                help="Page has <title> tag",
            )
            manual_has_favicon = st.checkbox(
                "Has Favicon",
                value=bool(quick_features.get("HasFavicon", 0))
                if "HasFavicon" in quick_features
                else None,
                help="Site has favicon",
            )
            manual_has_description = st.checkbox(
                "Has Meta Description",
                value=bool(quick_features.get("HasDescription", 0))
                if "HasDescription" in quick_features
                else None,
                help="Page has meta description",
            )
            manual_has_social = st.checkbox(
                "Has Social Media Links",
                value=bool(quick_features.get("HasSocialNet", 0))
                if "HasSocialNet" in quick_features
                else None,
                help="Links to social networks",
            )
            manual_has_copyright = st.checkbox(
                "Has Copyright Info",
                value=bool(quick_features.get("HasCopyrightInfo", 0))
                if "HasCopyrightInfo" in quick_features
                else None,
                help="Copyright notice present",
            )
            manual_is_responsive = st.checkbox(
                "Is Responsive",
                value=bool(quick_features.get("IsResponsive", 0))
                if "IsResponsive" in quick_features
                else None,
                help="Mobile-friendly design",
            )
            manual_robots = st.checkbox(
                "Has robots.txt",
                value=bool(quick_features.get("Robots", 0))
                if "Robots" in quick_features
                else None,
                help="robots.txt file exists",
            )

        with col_c:
            st.markdown("**Interactive Elements**")
            manual_no_of_popup = st.number_input(
                "Number of Popups",
                min_value=0,
                value=quick_features.get("NoOfPopup", None),
                help="Popup windows/modals",
            )
            manual_no_of_iframe = st.number_input(
                "Number of iFrames",
                min_value=0,
                value=quick_features.get("NoOfiFrame", None),
                help="Embedded iframes",
            )
            manual_has_submit = st.checkbox(
                "Has Submit Button",
                value=bool(quick_features.get("HasSubmitButton", 0))
                if "HasSubmitButton" in quick_features
                else None,
                help="Form submit buttons",
            )
            manual_has_hidden = st.checkbox(
                "Has Hidden Fields",
                value=bool(quick_features.get("HasHiddenFields", 0))
                if "HasHiddenFields" in quick_features
                else None,
                help="Hidden form inputs",
            )
            manual_has_password = st.checkbox(
                "Has Password Field",
                value=bool(quick_features.get("HasPasswordField", 0))
                if "HasPasswordField" in quick_features
                else None,
                help="Password input fields",
            )
            manual_has_ext_form = st.checkbox(
                "External Form Submit",
                value=bool(quick_features.get("HasExternalFormSubmit", 0))
                if "HasExternalFormSubmit" in quick_features
                else None,
                help="Form submits to external domain",
            )

        st.markdown("**References & Links**")
        col_d, col_e = st.columns(2)
        with col_d:
            manual_no_of_self_ref = st.number_input(
                "Internal Links",
                min_value=0,
                value=quick_features.get("NoOfSelfRef", None),
                help="Links to same domain",
            )
            manual_no_of_ext_ref = st.number_input(
                "External Links",
                min_value=0,
                value=quick_features.get("NoOfExternalRef", None),
                help="Links to other domains",
            )
        with col_e:
            manual_no_of_empty_ref = st.number_input(
                "Empty Links",
                min_value=0,
                value=quick_features.get("NoOfEmptyRef", None),
                help="Links with no href",
            )
            manual_no_of_redirect = st.number_input(
                "URL Redirects",
                min_value=0,
                value=quick_features.get("NoOfURLRedirect", None),
                help="Number of redirects",
            )

    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze URL", use_container_width=True, type="primary"
        )

    # Make prediction when button is clicked
    if analyze_button:
        if not url_input:
            st.warning("⚠️ Please enter a URL to analyze.")
        else:
            with st.spinner("🔄 Analyzing URL..."):
                # Start with quick_entry_features from CSV (if available)
                manual_features = st.session_state.get(
                    "quick_entry_features", {}
                ).copy()

                # Override with manual inputs from UI (if provided)
                if manual_line_of_code is not None:
                    manual_features["LineOfCode"] = manual_line_of_code
                if manual_largest_line is not None:
                    manual_features["LargestLineLength"] = manual_largest_line
                if manual_no_of_image is not None:
                    manual_features["NoOfImage"] = manual_no_of_image
                if manual_no_of_css is not None:
                    manual_features["NoOfCSS"] = manual_no_of_css
                if manual_no_of_js is not None:
                    manual_features["NoOfJS"] = manual_no_of_js
                if manual_has_title is not None:
                    manual_features["HasTitle"] = 1 if manual_has_title else 0
                if manual_has_favicon is not None:
                    manual_features["HasFavicon"] = 1 if manual_has_favicon else 0
                if manual_has_description is not None:
                    manual_features["HasDescription"] = (
                        1 if manual_has_description else 0
                    )
                if manual_has_social is not None:
                    manual_features["HasSocialNet"] = 1 if manual_has_social else 0
                if manual_has_copyright is not None:
                    manual_features["HasCopyrightInfo"] = (
                        1 if manual_has_copyright else 0
                    )
                if manual_is_responsive is not None:
                    manual_features["IsResponsive"] = 1 if manual_is_responsive else 0
                if manual_robots is not None:
                    manual_features["Robots"] = 1 if manual_robots else 0
                if manual_no_of_popup is not None:
                    manual_features["NoOfPopup"] = manual_no_of_popup
                if manual_no_of_iframe is not None:
                    manual_features["NoOfiFrame"] = manual_no_of_iframe
                if manual_has_submit is not None:
                    manual_features["HasSubmitButton"] = 1 if manual_has_submit else 0
                if manual_has_hidden is not None:
                    manual_features["HasHiddenFields"] = 1 if manual_has_hidden else 0
                if manual_has_password is not None:
                    manual_features["HasPasswordField"] = (
                        1 if manual_has_password else 0
                    )
                if manual_has_ext_form is not None:
                    manual_features["HasExternalFormSubmit"] = (
                        1 if manual_has_ext_form else 0
                    )
                if manual_no_of_self_ref is not None:
                    manual_features["NoOfSelfRef"] = manual_no_of_self_ref
                if manual_no_of_ext_ref is not None:
                    manual_features["NoOfExternalRef"] = manual_no_of_ext_ref
                if manual_no_of_empty_ref is not None:
                    manual_features["NoOfEmptyRef"] = manual_no_of_empty_ref
                if manual_no_of_redirect is not None:
                    manual_features["NoOfURLRedirect"] = manual_no_of_redirect

                result = predict_phishing(
                    url_input,
                    st.session_state.model,
                    st.session_state.scaler,
                    st.session_state.metadata,
                    manual_features,
                )

                # Clear quick entry after analysis
                if "quick_entry_url" in st.session_state:
                    del st.session_state.quick_entry_url
                if "quick_entry_features" in st.session_state:
                    del st.session_state.quick_entry_features

            if result:
                st.markdown("---")
                st.subheader("📊 Analysis Results")

                # Display result
                # IMPORTANT: prediction==0 means Phishing, prediction==1 means Legitimate
                if result["prediction"] == 0:
                    # Phishing detected (class 0)
                    st.markdown(
                        """
                    <div class="prediction-box phishing">
                        ⚠️ PHISHING DETECTED ⚠️<br>
                        This URL appears to be malicious!
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    st.error("""
                    **🚨 Security Alert:**
                    - Do NOT click on this link
                    - Do NOT enter any personal information
                    - Report this URL to your IT security team
                    """)
                else:
                    # Legitimate (class 1)
                    st.markdown(
                        """
                    <div class="prediction-box legitimate">
                        ✅ LEGITIMATE<br>
                        This URL appears to be safe
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    st.success("""
                    **✅ This URL appears safe, but:**
                    - Always verify the source
                    - Check for HTTPS encryption
                    - Be cautious with personal information
                    """)

                # Confidence metrics
                st.markdown("### 📈 Confidence Levels")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Phishing Probability",
                        f"{result['phishing_probability']:.1%}",
                        delta=None,
                    )
                    st.progress(result["phishing_probability"])

                with col2:
                    st.metric(
                        "Legitimate Probability",
                        f"{result['legitimate_probability']:.1%}",
                        delta=None,
                    )
                    st.progress(result["legitimate_probability"])

                st.info(f"""
                **Model Confidence:** {result["confidence"]:.1%}
                
                The model is {result["confidence"]:.1%} confident in this prediction.
                """)

    # Example URLs section
    with st.expander("📋 Try Example URLs"):
        st.markdown("""
        Click on these examples to test the detector:
        
        **Legitimate Examples:**
        - https://www.google.com
        - https://www.github.com
        - https://www.wikipedia.org
        
        **Suspicious Patterns to Watch For:**
        - Misspelled domains (gooogle.com, faceboook.com)
        - Unusual TLDs (.tk, .ml, .ga)
        - Long, complex URLs with many subdomains
        - URLs with @ symbols or encoded characters
        
        *Note: The actual classification depends on the extracted features*
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div class="footer">
        🔒 <strong>Phishing URL Detector</strong> | 
        Powered by Machine Learning | 
        CyberSecure Solutions Ltd.<br>
        <em>Always verify URLs before clicking. Stay safe online!</em>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
