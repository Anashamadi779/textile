import requests
import streamlit as st
from PIL import Image

DEFAULT_API_URL = "http://127.0.0.1:8001/analyze"

st.set_page_config(
    page_title="QualityAgent Dashboard",
    page_icon="🧵",
    layout="centered",
)

st.title("QualityAgent — Visual Quality Control")
st.markdown(
    "Upload a fabric image, analyze it via `POST /analyze`, "
    "and view detected defects with confidence scores plus a final conformity decision."
)

api_url = st.sidebar.text_input(
    "API URL",
    value=DEFAULT_API_URL,
    help="Change this if your API is running on a different host or port.",
)

uploaded_file = st.file_uploader(
    "Upload a fabric image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

conf = st.sidebar.slider("Confidence threshold", 0.01, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("IoU threshold", 0.01, 1.0, 0.45, 0.01)
st.sidebar.write("API endpoint: `http://127.0.0.1:8000/analyze`")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Unable to read the image: {exc}")
        st.stop()

    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Analyze"):
        with st.spinner("Calling QualityAgent API..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                }
                params = {"conf": conf, "iou": iou}
                response = requests.post(api_url, files=files, params=params, timeout=30)
            except requests.RequestException as exc:
                st.error(f"Unable to reach the API: {exc}")
                st.stop()

            if response.status_code != 200:
                st.error(
                    f"API error {response.status_code}: {response.text or response.reason}"
                )
                st.stop()

            data = response.json()

        nb_defauts = data.get("nb_defauts", 0)
        defauts = data.get("defauts", [])

        if nb_defauts > 0:
            st.markdown("### Final decision")
            st.error(f"NON-CONFORME — {nb_defauts} défaut(s) détecté(s)")
        else:
            st.markdown("### Final decision")
            st.success("CONFORME — Aucun défaut détecté")

        st.markdown("### Analysis results")
        st.write(f"**Image:** {data.get('image', uploaded_file.name)}")
        st.write(
            f"**Dimensions:** {data.get('largeur_px', '?')} x {data.get('hauteur_px', '?')} px"
        )
        st.write(f"**Nombre de défauts:** {nb_defauts}")

        if defauts:
            for idx, defect in enumerate(defauts, start=1):
                with st.expander(f"{idx}. {defect.get('classe', 'unknown')}"):
                    st.write(f"**Confiance:** {defect.get('confiance', 0.0):.4f}")
                    bbox = defect.get("bbox", {})
                    st.write(
                        f"**BBox:** x1={bbox.get('x1')} y1={bbox.get('y1')} "
                        f"x2={bbox.get('x2')} y2={bbox.get('y2')}"
                    )
        else:
            st.info("No defects were detected in this image.")

        with st.expander("Raw API response JSON"):
            st.json(data)
