import streamlit as st
from PIL import Image
from src.inference import predict_image
from main import load_config


def main():
    st.title("Chest X-ray Pneumonia Classifier")

    uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])
    config = load_config()

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", width=300)
        if st.button("Predict"):
            label = predict_image(image, config)
            st.write(f"**Prediction:** {label}")


if __name__ == "__main__":
    main()
