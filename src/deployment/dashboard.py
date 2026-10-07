# src/deployment/dashboard.py
# ============================================================
# CRIMINAL FACE RECOGNITION — FINAL DASHBOARD
# ============================================================

import streamlit as st
import numpy as np
import cv2
import os
import sys
import warnings

from PIL import Image

warnings.filterwarnings("ignore")

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.data_handling import CriminalFaceDataset

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Criminal Face Recognition",
    page_icon="👮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%); }
    .main-title { text-align: center; font-size: 2.4rem; font-weight: 700; color: #1a237e; padding: 10px 0 5px 0; margin: 0; }
    .sub-title { text-align: center; color: #666; font-size: 0.9rem; letter-spacing: 2px; margin-top: -5px; margin-bottom: 20px; }
    .card { background: #ffffff; border-radius: 14px; padding: 22px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); border: 1px solid #e8ecf1; margin: 8px 0; }
    .card-title { color: #1a237e; font-size: 1.1rem; font-weight: 600; margin-bottom: 12px; }
    .stButton > button { background: linear-gradient(135deg, #1a237e 0%, #283593 100%) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 10px 25px !important; font-weight: 600 !important; box-shadow: 0 3px 12px rgba(26, 35, 126, 0.2) !important; }
    .stButton > button:hover { transform: translateY(-2px) !important; }
    .divider { border: none; height: 1px; background: linear-gradient(90deg, transparent, #e8ecf1, transparent); margin: 25px 0; }
    .footer { text-align: center; color: #aaa; padding: 20px 0; font-size: 0.75rem; border-top: 1px solid #e8ecf1; margin-top: 30px; }
    .status-success { color: #27ae60; font-weight: 500; }
    .status-warning { color: #f39c12; font-weight: 500; }
    .police-logo { position: fixed; top: 15px; right: 20px; z-index: 1000; background: white; padding: 6px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DASHBOARD CLASS
# ============================================================

class CriminalDashboard:

    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self._init_session()

    def _init_session(self):
        keys = [
            "page", "model_loaded", "dataset_loaded", "prediction_result",
            "X", "y", "X_train_pca", "X_test_pca", "y_train", "y_test",
            "scaler", "pca", "model", "nn_model", "distance_threshold"
        ]
        for key in keys:
            if key not in st.session_state:
                if key == "page":
                    st.session_state[key] = "Home"
                elif key in ["model_loaded", "dataset_loaded"]:
                    st.session_state[key] = False
                else:
                    st.session_state[key] = None

    # ========================================================
    # LOAD DATASET — FAST
    # ========================================================
    def load_dataset(self):
        try:
            with st.spinner("Loading database..."):
                dataset = CriminalFaceDataset("data/criminal_faces")
                X, y = dataset.load_dataset()

                if X is None or y is None or len(X) == 0:
                    st.error("❌ No images found.")
                    return

                st.session_state.X = X
                st.session_state.y = y

                # FAST: 500 per class
                criminal_idx = np.where(y == 1)[0][:500]
                non_criminal_idx = np.where(y == 0)[0][:500]

                balanced = np.concatenate([criminal_idx, non_criminal_idx])
                np.random.shuffle(balanced)

                X_bal = X[balanced]
                y_bal = y[balanced]

                X_train, X_test, y_train, y_test = train_test_split(
                    X_bal, y_bal, test_size=0.20, random_state=42, stratify=y_bal
                )

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # FAST: fixed 50 PCA components
                pca = PCA(n_components=50, random_state=42)
                X_train_pca = pca.fit_transform(X_train_scaled)
                X_test_pca = pca.transform(X_test_scaled)

                nn_model = NearestNeighbors(n_neighbors=3, metric='euclidean')
                nn_model.fit(X_train_pca)

                train_dists, _ = nn_model.kneighbors(X_train_pca)
                avg_train_dists = np.mean(train_dists[:, 1:], axis=1)
                distance_threshold = float(np.percentile(avg_train_dists, 97))

                st.session_state.dataset_loaded = True
                st.session_state.X_train_pca = X_train_pca
                st.session_state.X_test_pca = X_test_pca
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.scaler = scaler
                st.session_state.pca = pca
                st.session_state.nn_model = nn_model
                st.session_state.distance_threshold = distance_threshold
                st.session_state.model_loaded = False
                st.session_state.prediction_result = None

            st.success("✅ Database Loaded")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Dataset error: {e}")

    # ========================================================
    # LOAD MODEL — FAST
    # ========================================================
    def load_model(self):
        try:
            X_train = st.session_state.X_train_pca
            y_train = st.session_state.y_train

            if X_train is None:
                st.warning("⚠️ Load database first!")
                return

            weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = dict(zip(np.unique(y_train), weights))

            with st.spinner("Training..."):
                # FAST: only Logistic Regression
                self.model = LogisticRegression(
                    max_iter=500, C=1.0,
                    class_weight=class_weight_dict,
                    random_state=42, n_jobs=-1
                )
                self.model.fit(X_train, y_train)

                st.session_state.model_loaded = True
                st.session_state.model = self.model

            st.success("✅ Model Loaded")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Model error: {e}")

    # ========================================================
    # PREDICT FACE
    # ========================================================
    def predict_face(self, image):
        try:
            if not st.session_state.model_loaded:
                return {"error": "Model not loaded"}

            model = st.session_state.model
            scaler = st.session_state.scaler
            pca = st.session_state.pca
            nn_model = st.session_state.get("nn_model")
            distance_threshold = st.session_state.get("distance_threshold", 30.0)

            img = np.array(image)
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            img = cv2.resize(img, (128, 128))
            flat = img.flatten().astype(np.float32) / 255.0

            flat_scaled = scaler.transform([flat])
            flat_pca = pca.transform(flat_scaled)

            if nn_model is not None:
                dists, _ = nn_model.kneighbors(flat_pca)
                avg_dist = float(np.mean(dists))
                if avg_dist > distance_threshold:
                    return {
                        "label": "Unknown",
                        "confidence": 0.0,
                        "prediction": -1,
                        "reason": "It is not found in any databases",
                        "database": "Not Found",
                        "action": "Requires Police Intervention"
                    }

            pred = int(model.predict(flat_pca)[0])
            proba = model.predict_proba(flat_pca)[0]
            confidence = float(max(proba))

            if confidence < 0.55:
                return {
                    "label": "Unknown",
                    "confidence": confidence,
                    "prediction": -1,
                    "reason": "It is not found in any databases",
                    "database": "Not Found",
                    "action": "Requires Police Intervention"
                }

            if pred == 1:
                return {
                    "label": "Criminal",
                    "confidence": confidence,
                    "prediction": 1,
                    "reason": "Found in Criminal Database",
                    "database": "Criminal Database",
                    "action": "Flag for Investigation"
                }
            else:
                return {
                    "label": "Non-Criminal",
                    "confidence": confidence,
                    "prediction": 0,
                    "reason": "Found in Non-Criminal Database",
                    "database": "Non-Criminal Database",
                    "action": "No Action Required"
                }

        except Exception as e:
            return {"error": str(e)}

    # ========================================================
    # RUN
    # ========================================================
    def run(self):
        self._render_sidebar()
        pages = {
            "Home": self._render_home,
            "Prediction": self._render_prediction,
            "Database Records": self._render_records,
            "Explainability": self._render_explainability
        }
        pages.get(st.session_state.page, self._render_home)()
        st.markdown("""
        <div class="footer">
            👮 Criminal Face Recognition System v4.0 &nbsp;|&nbsp; 🔒 For Law Enforcement
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # SIDEBAR
    # ========================================================
    def _render_sidebar(self):
        st.sidebar.markdown("""
        <div style="text-align:center;padding:10px 0;">
            <span style="font-size:2.5rem;">👮</span>
            <h3 style="color:#1a237e;margin:0;">Criminal</h3>
            <h3 style="color:#1a237e;margin:0;">Recognition</h3>
        </div>
        """, unsafe_allow_html=True)
        st.sidebar.markdown("---")

        pages = ["🏠 Home", "👤 Prediction", "📋 Database Records", "🔍 Explainability"]
        selected = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")
        page_map = {"🏠 Home": "Home", "👤 Prediction": "Prediction",
                    "📋 Database Records": "Database Records", "🔍 Explainability": "Explainability"}
        st.session_state.page = page_map.get(selected, "Home")

        st.sidebar.markdown("---")
        if st.session_state.dataset_loaded:
            st.sidebar.markdown('<span class="status-success">✅ Database Loaded</span>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<span class="status-warning">⚠️ Database Not Loaded</span>', unsafe_allow_html=True)

        if st.session_state.model_loaded:
            st.sidebar.markdown('<span class="status-success">✅ Model Loaded</span>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<span class="status-warning">⚠️ Model Not Loaded</span>', unsafe_allow_html=True)

        st.sidebar.markdown("---")

        if not st.session_state.dataset_loaded:
            if st.sidebar.button("📊 Load Database", use_container_width=True):
                self.load_dataset()

        if st.session_state.dataset_loaded and not st.session_state.model_loaded:
            if st.sidebar.button("🤖 Load Model", use_container_width=True):
                self.load_model()

        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="text-align:center;color:#aaa;font-size:0.7rem;letter-spacing:2px;">
            🔒 SECURE SYSTEM
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # HOME — FLASH CARDS
    # ========================================================
    def _render_home(self):
        st.markdown('<h1 class="main-title">👮 Criminal Face Recognition</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">AI-Powered Investigation Support System</p>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        X = st.session_state.X
        y = st.session_state.y
        loaded = st.session_state.dataset_loaded and X is not None

        total = f"{len(X):,}" if loaded else "---"
        criminal = f"{int(np.sum(y == 1)):,}" if loaded else "---"
        non_criminal = f"{int(np.sum(y == 0)):,}" if loaded else "---"

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f'''
            <div style="background:#ffffff;border-radius:20px;padding:35px 20px;text-align:center;box-shadow:0 10px 30px rgba(0,0,0,0.08);border-top:6px solid #1a237e;">
                <div style="font-size:3.2rem;margin-bottom:12px;">📊</div>
                <div style="font-size:2.6rem;font-weight:800;color:#1a237e;margin-bottom:6px;">{total}</div>
                <div style="font-size:0.85rem;font-weight:600;color:#777;letter-spacing:1.5px;text-transform:uppercase;">Total Samples</div>
            </div>
            ''', unsafe_allow_html=True)

        with c2:
            st.markdown(f'''
            <div style="background:#ffffff;border-radius:20px;padding:35px 20px;text-align:center;box-shadow:0 10px 30px rgba(0,0,0,0.08);border-top:6px solid #e74c3c;">
                <div style="font-size:3.2rem;margin-bottom:12px;">🚨</div>
                <div style="font-size:2.6rem;font-weight:800;color:#e74c3c;margin-bottom:6px;">{criminal}</div>
                <div style="font-size:0.85rem;font-weight:600;color:#777;letter-spacing:1.5px;text-transform:uppercase;">Criminal</div>
            </div>
            ''', unsafe_allow_html=True)

        with c3:
            st.markdown(f'''
            <div style="background:#ffffff;border-radius:20px;padding:35px 20px;text-align:center;box-shadow:0 10px 30px rgba(0,0,0,0.08);border-top:6px solid #27ae60;">
                <div style="font-size:3.2rem;margin-bottom:12px;">✅</div>
                <div style="font-size:2.6rem;font-weight:800;color:#27ae60;margin-bottom:6px;">{non_criminal}</div>
                <div style="font-size:0.85rem;font-weight:600;color:#777;letter-spacing:1.5px;text-transform:uppercase;">Non-Criminal</div>
            </div>
            ''', unsafe_allow_html=True)

    # ========================================================
    # PREDICTION
    # ========================================================
    def _render_prediction(self):
        st.markdown('<h1 class="main-title">👤 Face Identification</h1>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        if not st.session_state.dataset_loaded:
            st.warning("⚠️ Load the database first from the sidebar.")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card"><div class="card-title">📤 Upload Image</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)
                if st.button("🔍 Identify Face", type="primary", use_container_width=True):
                    with st.spinner("Comparing..."):
                        result = self.predict_face(image)
                        st.session_state.prediction_result = result
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card"><div class="card-title">📊 Result</div>', unsafe_allow_html=True)
            result = st.session_state.prediction_result
            if result and not result.get("error"):
                label = result.get("label", "Unknown")
                confidence = result.get("confidence", 0)
                reason = result.get("reason", "")

                if label == "Criminal":
                    bg, border, color, icon = "#ffe8e8", "#e74c3c", "#e74c3c", "🚨"
                elif label == "Non-Criminal":
                    bg, border, color, icon = "#e0ffe8", "#27ae60", "#27ae60", "✅"
                else:
                    bg, border, color, icon = "#fff3d6", "#f39c12", "#f39c12", "❓"

                st.markdown(f'''
                <div style="background:{bg};border:3px solid {border};border-radius:20px;padding:35px 25px;text-align:center;box-shadow:0 12px 35px rgba(0,0,0,0.12);">
                    <div style="font-size:4.5rem;margin-bottom:10px;">{icon}</div>
                    <div style="font-size:2.2rem;font-weight:800;color:{color};letter-spacing:2px;margin-bottom:10px;">{label.upper()}</div>
                    <div style="font-size:1rem;color:#555;font-style:italic;">{reason}</div>
                </div>
                ''', unsafe_allow_html=True)

                st.markdown(f'''
                <div style="background:linear-gradient(135deg,#e3f2fd,#bbdefb);border-radius:18px;padding:22px;text-align:center;border:2px solid #90caf9;margin-top:15px;box-shadow:0 8px 25px rgba(13,71,161,0.15);">
                    <div style="font-size:0.85rem;font-weight:600;color:#0d47a1;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;">Match Confidence</div>
                    <div style="font-size:2.6rem;font-weight:800;color:#0d47a1;">{confidence:.2%}</div>
                </div>
                ''', unsafe_allow_html=True)
            elif result and result.get("error"):
                st.error(f"❌ {result['error']}")
            else:
                st.info("ℹ️ Upload an image and click 'Identify Face'.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================
    # DATABASE RECORDS
    # ========================================================
    def _render_records(self):
        st.markdown('<h1 class="main-title">📋 Database Records</h1>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        if not st.session_state.dataset_loaded:
            st.warning("⚠️ Load database first!")
            return

        y = st.session_state.y
        total = len(y)
        criminal = int(np.sum(y == 1))
        non_criminal = int(np.sum(y == 0))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'''
            <div style="background:#ffffff;border-radius:20px;padding:30px;text-align:center;box-shadow:0 10px 30px rgba(0,0,0,0.08);border-top:6px solid #1a237e;">
                <div style="font-size:2.6rem;font-weight:800;color:#1a237e;">{total:,}</div>
                <div style="font-size:0.85rem;font-weight:600;color:#777;letter-spacing:1.5px;text-transform:uppercase;margin-top:6px;">Total Records</div>
            </div>
            ''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''
            <div style="background:#ffffff;border-radius:20px;padding:30px;text-align:center;box-shadow:0 10px 30px rgba(0,0,0,0.08);border-top:6px solid #e74c3c;">
                <div style="font-size:2.6rem;font-weight:800;color:#e74c3c;">{criminal:,}</div>
                <div style="font-size:0.85rem;font-weight:600;color:#777;letter-spacing:1.5px;text-transform:uppercase;margin-top:6px;">Criminal Records</div>
            </div>
            ''', unsafe_allow_html=True)
        with c3:
            st.markdown(f'''
            <div style="background:#ffffff;border-radius:20px;padding:30px;text-align:center;box-shadow:0 10px 30px rgba(0,0,0,0.08);border-top:6px solid #27ae60;">
                <div style="font-size:2.6rem;font-weight:800;color:#27ae60;">{non_criminal:,}</div>
                <div style="font-size:0.85rem;font-weight:600;color:#777;letter-spacing:1.5px;text-transform:uppercase;margin-top:6px;">Non-Criminal Records</div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🔍 Search Records</div>', unsafe_allow_html=True)
        search_type = st.selectbox("Filter by Category", ["All Records", "Criminal Only", "Non-Criminal Only"])
        num_records = st.slider("Number of records to display", 5, 50, 10)

        if search_type == "Criminal Only":
            indices = np.where(y == 1)[0][:num_records]
        elif search_type == "Non-Criminal Only":
            indices = np.where(y == 0)[0][:num_records]
        else:
            indices = np.arange(min(num_records, total))

        st.markdown(f"**Showing {len(indices)} records**")

        for idx in indices:
            lab = "Criminal" if y[idx] == 1 else "Non-Criminal"
            color = "#e74c3c" if y[idx] == 1 else "#27ae60"
            status = "Flagged" if y[idx] == 1 else "Verified"
            st.markdown(f'''
            <div style="background:#ffffff;border-radius:14px;padding:18px 22px;margin:10px 0;border-left:6px solid {color};box-shadow:0 4px 14px rgba(0,0,0,0.05);">
                <div style="font-weight:700;color:#1a237e;font-size:1rem;margin-bottom:4px;">Record #{idx + 1} — {lab}</div>
                <div style="color:#666;font-size:0.85rem;">Database ID: DB-{idx + 1000} &nbsp;|&nbsp; Status: {status}</div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================
    # EXPLAINABILITY
    # ========================================================
    def _render_explainability(self):
        st.markdown('<h1 class="main-title">🔍 Explainable AI (XAI)</h1>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        if not st.session_state.dataset_loaded:
            st.warning("⚠️ Load database first!")
            return
        if st.session_state.prediction_result is None:
            st.info("ℹ️ First, go to the Prediction page and identify a face.")
            return

        result = st.session_state.prediction_result
        label = result.get("label", "Unknown")
        confidence = result.get("confidence", 0)
        reason = result.get("reason", "")
        database = result.get("database", "N/A")
        action = result.get("action", "N/A")

        if label == "Criminal":
            bg, border, color, icon = "#ffe8e8", "#e74c3c", "#e74c3c", "🚨"
            interpretation = "The uploaded face matches an entry in the <b>Criminal Database</b>. This individual is flagged as a person of interest. Proceed with standard investigation protocols."
        elif label == "Non-Criminal":
            bg, border, color, icon = "#e0ffe8", "#27ae60", "#27ae60", "✅"
            interpretation = "The uploaded face matches an entry in the <b>Non-Criminal Database</b>. This individual has no criminal record. No immediate action is required."
        else:
            bg, border, color, icon = "#fff3d6", "#f39c12", "#f39c12", "❓"
            interpretation = "The uploaded face does <b>not match any entry</b> in either database. This individual is unknown to the system. <b>Police intervention is recommended.</b>"

        st.markdown(f'''
        <div style="background:{bg};border:3px solid {border};border-radius:20px;padding:35px 25px;text-align:center;box-shadow:0 12px 35px rgba(0,0,0,0.12);">
            <div style="font-size:4.5rem;margin-bottom:10px;">{icon}</div>
            <div style="font-size:2.2rem;font-weight:800;color:{color};letter-spacing:2px;margin-bottom:10px;">{label.upper()}</div>
            <div style="font-size:1rem;color:#555;font-style:italic;">{reason}</div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
        <div style="background:#ffffff;border-radius:16px;padding:22px 24px;box-shadow:0 6px 20px rgba(0,0,0,0.06);border-left:6px solid {border};margin:16px 0;">
            <div style="color:#1a237e;font-size:1.05rem;font-weight:700;margin-bottom:14px;">🧾 Explanation Summary</div>
            <div style="display:flex;justify-content:space-between;padding:12px 16px;background:#f8f9fa;border-radius:10px;margin:8px 0;">
                <span style="color:#555;font-weight:500;">Prediction</span>
                <span style="color:{color};font-weight:700;">{label}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:12px 16px;background:#f8f9fa;border-radius:10px;margin:8px 0;">
                <span style="color:#555;font-weight:500;">Confidence Score</span>
                <span style="color:#1a237e;font-weight:700;">{confidence:.2%}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:12px 16px;background:#f8f9fa;border-radius:10px;margin:8px 0;">
                <span style="color:#555;font-weight:500;">Matched Database</span>
                <span style="color:#1a237e;font-weight:700;">{database}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:12px 16px;background:#f8f9fa;border-radius:10px;margin:8px 0;">
                <span style="color:#555;font-weight:500;">Reason</span>
                <span style="color:#1a237e;font-weight:700;">{reason}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:12px 16px;background:#f8f9fa;border-radius:10px;margin:8px 0;">
                <span style="color:#555;font-weight:500;">Recommended Action</span>
                <span style="color:#1a237e;font-weight:700;">{action}</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
        <div style="background:#ffffff;border-radius:16px;padding:22px 24px;box-shadow:0 6px 20px rgba(0,0,0,0.06);border-left:6px solid {border};margin:12px 0;">
            <div style="color:#1a237e;font-size:1.05rem;font-weight:700;margin-bottom:14px;">📌 Interpretation</div>
            <p style="color:#333;line-height:1.7;">{interpretation}</p>
        </div>
        ''', unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================

def main():
    st.markdown("""
    <div class="police-logo">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Indian_Police_Logo.svg/100px-Indian_Police_Logo.svg.png" width="60">
    </div>
    """, unsafe_allow_html=True)

    dashboard = CriminalDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
