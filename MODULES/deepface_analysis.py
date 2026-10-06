
import streamlit as st
import numpy as np
from PIL import Image
from deepface import DeepFace
import tempfile
import os


def deepface_analysis_page():

    st.title("🧠 AI FACE INSIGHT")
    st.subheader("DeepFace Facial Analysis Engine")
    st.caption("AI • DEEP LEARNING • FACE ANALYSIS")

    st.divider()

    uploaded_file = st.file_uploader(
        "📸 Upload a Face Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("🖼️ Input Image")

        st.image(
            image,
            caption="Uploaded Face Image",
            width="stretch"
        )

        st.write("")

        analyze_button = st.button(
            "🚀 ANALYZE FACE",
            width="stretch"
        )

        if analyze_button:

            with st.spinner(
                "🧠 DeepFace is analyzing the face..."
            ):

                temp_path = None

                try:

                    # Create temporary image file
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".jpg"
                    ) as temp_file:

                        image.save(
                            temp_file,
                            format="JPEG"
                        )

                        temp_path = temp_file.name

                    # DeepFace emotion analysis
                    result = DeepFace.analyze(
                        img_path=temp_path,
                        actions=["emotion"],
                        enforce_detection=False
                    )

                    # Handle different DeepFace return formats
                    if isinstance(result, list):
                        result = result[0]

                    dominant_emotion = result.get(
                        "dominant_emotion",
                        "Unknown"
                    )

                    emotion_scores = result.get(
                        "emotion",
                        {}
                    )

                except Exception as e:

                    st.error(
                        "❌ DeepFace analysis failed."
                    )

                    st.caption(str(e))
                    return

                finally:

                    if (
                        temp_path
                        and os.path.exists(temp_path)
                    ):
                        os.remove(temp_path)

            st.success(
                "✅ Face analysis completed successfully!"
            )

            st.divider()

            st.subheader("📊 Analysis Result")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "😊 DOMINANT EMOTION",
                    dominant_emotion.upper()
                )

            with col2:

                if emotion_scores:

                    confidence = max(
                        emotion_scores.values()
                    )

                    st.metric(
                        "🎯 CONFIDENCE",
                        f"{confidence:.1f}%"
                    )

                else:

                    st.metric(
                        "🎯 CONFIDENCE",
                        "N/A"
                    )

            st.divider()

            st.subheader("🧠 Emotion Analysis")

            if emotion_scores:

                for emotion, score in emotion_scores.items():

                    st.write(
                        f"**{emotion.capitalize()}** "
                        f"— {score:.1f}%"
                    )

                    st.progress(
                        min(int(score), 100)
                    )

            st.divider()

            with st.expander(
                "🔬 Technical Details"
            ):

                st.write(
                    "Model: DeepFace"
                )

                st.write(
                    "Analysis Type: Facial Emotion Analysis"
                )

                st.write(
                    "Method: Deep Learning"
                )

                st.write(
                    "Detection: Automatic Face Detection"
                )

    else:

        st.info(
            "📸 Upload a face image to activate "
            "the AI Face Insight engine."
        )

