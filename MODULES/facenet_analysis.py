
import streamlit as st
import numpy as np
from PIL import Image
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine


def facenet_analysis_page():

    st.title("📐 FACE SIMILARITY LAB")
    st.subheader("FaceNet Similarity Engine")
    st.caption("AI • FACE EMBEDDINGS • SIMILARITY ANALYSIS")

    st.divider()

    st.write(
        "Upload two face images to compare their "
        "facial feature similarity."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("👤 FACE 01")

        image1_file = st.file_uploader(
            "Upload First Face",
            type=["jpg", "jpeg", "png"],
            key="face1"
        )

        if image1_file is not None:

            image1 = Image.open(
                image1_file
            ).convert("RGB")

            st.image(
                image1,
                caption="Face 01",
                width="stretch"
            )

    with col2:

        st.subheader("👤 FACE 02")

        image2_file = st.file_uploader(
            "Upload Second Face",
            type=["jpg", "jpeg", "png"],
            key="face2"
        )

        if image2_file is not None:

            image2 = Image.open(
                image2_file
            ).convert("RGB")

            st.image(
                image2,
                caption="Face 02",
                width="stretch"
            )

    st.write("")

    compare_button = st.button(
        "🚀 COMPARE FACES",
        width="stretch"
    )

    if compare_button:

        if image1_file is None or image2_file is None:

            st.warning(
                "⚠️ Please upload both face images."
            )

            return

        with st.spinner(
            "🧠 FaceNet is generating facial embeddings..."
        ):

            try:

                # Convert images to NumPy arrays
                img1 = np.array(image1)
                img2 = np.array(image2)

                # Load FaceNet model
                embedder = FaceNet()

                # Generate embeddings
                embedding1 = embedder.embeddings(
                    [img1]
                )[0]

                embedding2 = embedder.embeddings(
                    [img2]
                )[0]

                # Calculate cosine distance
                distance = cosine(
                    embedding1,
                    embedding2
                )

                # Convert distance to similarity
                similarity = (
                    1 - distance
                ) * 100

                similarity = max(
                    0,
                    min(100, similarity)
                )

            except Exception as e:

                st.error(
                    "❌ FaceNet comparison failed."
                )

                st.caption(str(e))

                return

        st.success(
            "✅ Face comparison completed!"
        )

        st.divider()

        st.subheader(
            "📊 Similarity Result"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "📐 SIMILARITY",
                f"{similarity:.1f}%"
            )

        with col2:

            st.metric(
                "📏 COSINE DISTANCE",
                f"{distance:.3f}"
            )

        with col3:

            st.metric(
                "🧠 EMBEDDING SIZE",
                str(len(embedding1))
            )

        st.divider()

        st.subheader(
            "🎯 Comparison Status"
        )

        st.progress(
            int(similarity)
        )

        if similarity >= 70:

            st.success(
                "🟢 HIGH FEATURE SIMILARITY"
            )

            st.write(
                "The two images show a high level "
                "of facial feature similarity."
            )

        elif similarity >= 50:

            st.warning(
                "🟡 MODERATE FEATURE SIMILARITY"
            )

            st.write(
                "The two images show moderate "
                "facial feature similarity."
            )

        else:

            st.info(
                "🔵 LOW FEATURE SIMILARITY"
            )

            st.write(
                "The two images show low "
                "facial feature similarity."
            )

        st.divider()

        with st.expander(
            "🔬 Technical Details"
        ):

            st.write(
                "Model: FaceNet"
            )

            st.write(
                "Feature Representation: Face Embeddings"
            )

            st.write(
                "Comparison Method: Cosine Distance"
            )

            st.write(
                "Output: Feature Similarity Score"
            )

