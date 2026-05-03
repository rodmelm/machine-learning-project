import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Despliegue del Modelo", page_icon="🚀", layout="centered")
st.title("Predicción con nuevos datos")
st.write("Esta aplicación utiliza un `Pipeline` completo de `scikit-learn`.")

@st.cache_resource
def load_pack():
    return joblib.load("modelo_final.pkl")

try:
    pipeline = load_pack()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

st.markdown("### Introduce los valores de las variables:")

# Usamos un formulario para evitar recalcular la predicción con cada pulsación
with st.form("prediction_form"):
    st.markdown("### Datos del Cliente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Variables Numéricas")
        age = st.number_input("Edad (age)", min_value=18, max_value=100, value=30)
        balance = st.number_input("Saldo (balance)", value=0)
        day = st.number_input("Día del último contacto (day)", min_value=1, max_value=31, value=15)
        duration = st.number_input("Duración de la llamada (duration)", min_value=0, value=150)
        campaign = st.number_input("Número de contactos durante la campaña (campaign)", min_value=1, value=1)
        pdays = st.number_input("Número de días desde el último contacto (pdays)", min_value=-1, value=1200)
        previous = st.number_input("Número de contactos realizados antes de esta campaña (previous)", min_value=0, value=0)
        
    with col2:
        st.subheader("Variables Categóricas")
        job = st.selectbox("Trabajo (job)", ["management", "blue-collar", "technician", "admin.", "services", "retired", "self-employed", "student", "unemployed", "entrepreneur", "housemaid", "unknown"])
        marital = st.selectbox("Estado Civil (marital)", ["divorced", "married", "single", "unknown"])
        education = st.selectbox("Educación (education)", ["primary", "secondary", "tertiary", "unknown"])
        default = st.selectbox("¿Tiene crédito en default? (default)", ["no", "yes", "unknown"])
        housing = st.selectbox("¿Tiene préstamo de vivienda? (housing)", ["no", "yes", "unknown"])
        loan = st.selectbox("¿Tiene préstamo personal? (loan)", ["no", "yes", "unknown"])
        contact = st.selectbox("Tipo de contacto (contact)", ["cellular", "telephone", "unknown"])
        month = st.selectbox("Mes del último contacto (month)", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
        poutcome = st.selectbox("Resultado de la campaña anterior (poutcome)", ["failure", "success", "other", "unknown"])
        
       
    st.markdown("---")
    submitted = st.form_submit_button("Predecir", use_container_width=True)

if submitted:
    # Convertimos las entradas en un DataFrame de una única fila
    X_new = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])
    
    try:
        # Predecimos usando el Pipeline (que ya incorpora todo el preprocesamiento)
        proba = pipeline.predict_proba(X_new)[0]
        y_pred = pipeline.predict(X_new)[0]
        classes_ = pipeline.classes_
        
        y_pred = "Sí Contrata" if y_pred == 1 else "No Contrata"

        st.success(f"### Predicción: **{y_pred}**")
        
        st.markdown("#### Probabilidades de la Predicción:")
        
        # Mostramos los resultados como métricas destacadas
        cols = st.columns(len(classes_))
        for i, (cls, p) in enumerate(zip(classes_, proba)):
            # Le ponemos un nombre más amigable que simplemente "0" o "1"
            nombre_clase = "Sí Contrata" if cls == 1 else "No Contrata"
            cols[i].metric(label=f"{nombre_clase}", value=f"{p*100:.1f}%")

    except Exception as e:
        st.error(f"Error al predecir. Revisa las variables. Detalles: {e}")