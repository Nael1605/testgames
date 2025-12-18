import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Configuration de la page
st.set_page_config(
    page_title="YOLO Real-Time AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour la couleur de fond
st.markdown("""
    <style>
        .main {
            background-color: #a9a9a9;
        }
        [data-testid="stSidebar"] {
            background-color: #1e1e1e;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 YOLO Real-Time AI")
st.write("Chargez une image pour détecter les objets avec YOLO")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    # Sélection du modèle
    model_name = st.selectbox(
        "Sélectionnez le modèle YOLO",
        ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
        help="Plus grand = meilleure précision mais plus lent"
    )
    
    # Confiance
    confidence = st.slider(
        "Confiance minimale",
        0.0, 1.0, 0.5,
        step=0.05,
        help="Seuil de confiance pour la détection"
    )
    
    # IOU
    iou = st.slider(
        "IOU Threshold",
        0.0, 1.0, 0.5,
        step=0.05,
        help="Seuil IoU pour la suppression des doublons"
    )

# Charger le modèle
@st.cache_resource
def load_model(model_name):
    model = YOLO(f"{model_name}.pt")
    return model

# Zone principale
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Image source")
    uploaded_file = st.file_uploader(
        "Chargez une image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Formats supportés: JPG, PNG, BMP, WebP"
    )

with col2:
    st.subheader("🏁 Résultats")
    results_placeholder = st.empty()

if uploaded_file is not None:
    # Afficher l'image source
    image = Image.open(uploaded_file)
    col1.image(image, use_container_width=True)
    
    # Charger le modèle et faire la détection
    with st.spinner(f"Chargement du modèle {model_name}..."):
        model = load_model(model_name)
    
    with st.spinner("Détection en cours..."):
        # Convertir l'image en array numpy
        img_array = np.array(image)
        
        # Lancer la détection
        results = model.predict(
            source=img_array,
            conf=confidence,
            iou=iou,
            verbose=False
        )
    
    # Afficher les résultats
    if results and len(results) > 0:
        result = results[0]
        
        # Dessiner les boîtes de détection
        annotated_image = result.plot()
        col2.image(annotated_image, use_container_width=True)
        
        # Afficher les détections
        if result.boxes is not None and len(result.boxes) > 0:
            st.subheader("📊 Détections")
            
            detections_data = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence_score = float(box.conf[0])
                class_name = result.names[class_id]
                
                x1, y1, x2, y2 = box.xyxy[0]
                
                detections_data.append({
                    "Objet": class_name,
                    "Confiance": f"{confidence_score:.2%}",
                    "X1": int(x1),
                    "Y1": int(y1),
                    "X2": int(x2),
                    "Y2": int(y2)
                })
            
            # Afficher sous forme de tableau
            st.dataframe(
                detections_data,
                use_container_width=True,
                hide_index=True
            )
            
            # Statistiques
            st.subheader("📈 Statistiques")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Nombre d'objets", len(detections_data))
            
            with col_stat2:
                avg_conf = np.mean([float(d["Confiance"].strip('%'))/100 for d in detections_data])
                st.metric("Confiance moyenne", f"{avg_conf:.2%}")
            
            with col_stat3:
                unique_objects = len(set(d["Objet"] for d in detections_data))
                st.metric("Types d'objets", unique_objects)
        else:
            st.warning("Aucun objet détecté avec ce seuil de confiance")
    else:
        st.error("Erreur lors de la détection")
else:
    st.info("👈 Chargez une image pour commencer")
