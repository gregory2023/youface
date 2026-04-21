import streamlit as st
import requests
import pandas as pd
import time

API_URL = "http://13.61.34.115:8000"

st.set_page_config(page_title="TFG Vision System", page_icon="👁", layout="wide")
st.title("TFG Vision System — Panel de Control")

# Refresco automático cada 10 segundos
refresh = st.empty()

def cargar_alertas():
    try:
        response = requests.get(f"{API_URL}/alertas", timeout=5)
        return response.json()
    except:
        st.error("No se puede conectar con la API")
        return []

alertas = cargar_alertas()

if alertas:
    df = pd.DataFrame(alertas)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total alertas", len(df))
    col2.metric("Detecciones faciales", len(df[df['alert_type'] == 'facial_detection']))
    col3.metric("Moviles detectados", len(df[df['alert_type'] == 'movil_detected']))

    st.subheader("Registro de alertas")
    st.dataframe(df[['timestamp', 'user_id', 'alert_type']].sort_values('timestamp', ascending=False), use_container_width=True)

    st.subheader("Alertas por tipo")
    st.bar_chart(df['alert_type'].value_counts())
else:
    st.info("No hay alertas registradas aun.")

st.caption(f"Ultima actualizacion: {time.strftime('%H:%M:%S')} — refresca cada 10s")
time.sleep(10)
st.rerun()
