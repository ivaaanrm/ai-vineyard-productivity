"""Agrixel Image Download & Explorer — Streamlit entry point."""

import streamlit as st

st.set_page_config(
    page_title="Agrixel — Satellite Image Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Agrixel — Satellite Image Explorer")

st.markdown(
    """
    Herramienta para la descarga y exploración de imágenes satelitales
    de parcelas vitícolas.

    ### Páginas

    - **Download** — Configura y lanza descargas de imágenes Sentinel‑1 / Sentinel‑2
      mediante el contenedor Docker de agrixel‑fair.
    - **Explorer** — Explora los datos descargados: mapas interactivos,
      series temporales y análisis de bandas.
    - **Processed Dataset** — Visualiza el dataset procesado por parcela,
      incluyendo métricas YoY, metadata y snapshots del datacube.

    ### Uso rápido

    1. Ve a la página **Download**, selecciona parcelas y rango de fechas,
       y pulsa *Descargar*.
    2. Cuando termine, pasa a **Explorer** para visualizar los resultados.
    3. Si ya tienes un dataset procesado, usa **Processed Dataset** para
       revisar métricas agregadas, target `yield_kg_ha` e imágenes.
    """
)
