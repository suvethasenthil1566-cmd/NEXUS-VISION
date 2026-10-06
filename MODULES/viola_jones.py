import streamlit as st
import cv2
import numpy as np
from PIL import Image


def viola_jones_page():

    # ==================================================
    # BRIGHT TEXT + FUTURISTIC STYLING
    # ==================================================

    st.markdown("""
    <style>

        /* Main headings */
        h1 {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        h2, h3 {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        /* Normal text */
        p {
            color: #d9e2ff !important;
        }

        /* Caption */
        .stCaption {
            color: #00f5ff !important;
        }

        /* File uploader label */
        label {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* Markdown text */
        .stMarkdown {
            color: #ffffff;
        }

        /* Metric value */
        [data-testid="stMetricValue"] {
            color: #00f5ff !important;
            font-weight: 800 !important;
        }

        /* Metric label */
        [data-testid="stMetricLabel"] {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* Alert text */
        .stAlert {
            color: #ffffff !important;
        }

        /* Button */
        .stButton > button {
            width: 100%;
            border-radius: 12px;
            padding: 12px;
            font-weight: 700;
        }

    </style>
    """, unsafe_allow_html=True)


    # ==================================================
    # HEADER
    # ==================================================

    st.title("👤 FACE SCANNER")

    st.subheader(
        "Viola–Jones Face Detection Engine"
    )

    st.caption(
        "AI • REAL-TIME VISION • FACE DETECTION"
    )

    st.divider()


    # ==================================================
    # IMAGE UPLOAD
    # ==================================================

    uploaded_file = st.file_uploader(
        "📸 Upload a Face Image",
        type=["jpg", "jpeg", "png"]
    )


    # ==================================================
    # IMAGE AVAILABLE
    # ==================================================

    if uploaded_file is not None:

        # Read image
        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_np = np.array(image)


        # ==================================================
        # INPUT IMAGE
        # ==================================================

        st.subheader("🖼️ Input Image")

        st.image(
            image,
            caption="Uploaded Face Image",
            use_container_width=True
        )

        st.write("")


        # ==================================================
        # SCAN BUTTON
        # ==================================================

        scan_button = st.button(
            "🚀 START FACE SCAN",
            use_container_width=True
        )


        if scan_button:

            # ==================================================
            # PROCESSING
            # ==================================================

            with st.spinner(
                "🔎 Scanning image for faces..."
            ):

                # RGB → BGR
                frame = cv2.cvtColor(
                    image_np,
                    cv2.COLOR_RGB2BGR
                )


                # BGR → Gray
                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )


                # ==================================================
                # HAAR CASCADE
                # ==================================================

                cascade_path = (
                    cv2.data.haarcascades
                    + "haarcascade_frontalface_default.xml"
                )


                face_cascade = cv2.CascadeClassifier(
                    cascade_path
                )


                # Check classifier
                if face_cascade.empty():

                    st.error(
                        "❌ Haar Cascade could not be loaded."
                    )

                    return


                # ==================================================
                # FACE DETECTION
                # ==================================================

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(50, 50)
                )


                # Copy image
                result = frame.copy()


                # ==================================================
                # DRAW FACE BOXES
                # ==================================================

                for i, (x, y, w, h) in enumerate(faces):

                    # Green rectangle
                    cv2.rectangle(
                        result,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        3
                    )


                    # Face label
                    cv2.putText(
                        result,
                        f"FACE {i + 1}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )


                # BGR → RGB
                result_rgb = cv2.cvtColor(
                    result,
                    cv2.COLOR_BGR2RGB
                )


            # ==================================================
            # SCAN COMPLETE
            # ==================================================

            st.success(
                "✅ Face scanning completed successfully!"
            )

            st.divider()


            # ==================================================
            # RESULTS
            # ==================================================

            st.subheader(
                "📊 Scan Results"
            )


            # ==================================================
            # METRICS
            # ==================================================

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "👤 FACES DETECTED",
                    len(faces)
                )


            with col2:

                st.metric(
                    "⚙️ SCALE FACTOR",
                    "1.1"
                )


            with col3:

                st.metric(
                    "🎯 MIN NEIGHBORS",
                    "5"
                )


            st.divider()


            # ==================================================
            # DETECTION OUTPUT
            # ==================================================

            st.subheader(
                "🧠 Detection Output"
            )


            st.image(
                result_rgb,
                caption="Viola–Jones Face Detection Result",
                use_container_width=True
            )


            # ==================================================
            # FACE LOCATIONS
            # ==================================================

            if len(faces) > 0:

                st.subheader(
                    "📍 Face Locations"
                )


                for i, (x, y, w, h) in enumerate(faces):

                    st.info(
                        f"👤 FACE {i + 1}  |  "
                        f"X: {x}  |  "
                        f"Y: {y}  |  "
                        f"Width: {w}  |  "
                        f"Height: {h}"
                    )


            else:

                st.warning(
                    "⚠️ No face detected. "
                    "Please try a clear front-facing image."
                )


            # ==================================================
            # TECHNICAL DETAILS
            # ==================================================

            with st.expander(
                "🔬 Technical Details"
            ):

                st.write(
                    "Algorithm: Viola–Jones"
                )

                st.write(
                    "Classifier: Haar Cascade"
                )

                st.write(
                    "Detection Method: detectMultiScale"
                )

                st.write(
                    "Scale Factor: 1.1"
                )

                st.write(
                    "Minimum Neighbors: 5"
                )

                st.write(
                    "Minimum Face Size: 50 × 50 pixels"
                )


    # ==================================================
    # NO IMAGE
    # ==================================================

    else:

        st.info(
            "📸 Upload a face image to activate "
            "the Face Scanner."
        )