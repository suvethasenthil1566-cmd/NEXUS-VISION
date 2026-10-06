
import streamlit as st
import cv2
import numpy as np
from PIL import Image


def template_matching_page():

    st.markdown(
        """
        <h1 style="text-align:center; color:#38bdf8;">
        🔍 OBJECT HUNT
        </h1>

        <p style="text-align:center; color:#cbd5e1; font-size:18px;">
        Find a specific object or pattern inside an image
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # -------------------------------------------------
    # Upload section
    # -------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🖼️ Main Image")
        main_file = st.file_uploader(
            "Upload the image to search",
            type=["jpg", "jpeg", "png"],
            key="main_image"
        )

    with col2:
        st.markdown("### 🎯 Target Template")
        template_file = st.file_uploader(
            "Upload the object/template to find",
            type=["jpg", "jpeg", "png"],
            key="template_image"
        )

    st.divider()

    # -------------------------------------------------
    # Show uploaded images
    # -------------------------------------------------

    if main_file and template_file:

        main_image = Image.open(main_file).convert("RGB")
        template_image = Image.open(template_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔎 Search Image")
            st.image(main_image, use_container_width=True)

        with col2:
            st.markdown("#### 🎯 Target Object")
            st.image(template_image, use_container_width=True)

        st.divider()

        # -------------------------------------------------
        # Scan button
        # -------------------------------------------------

        if st.button(
            "🚀 SCAN FOR MATCH",
            use_container_width=True,
            type="primary"
        ):

            # Convert PIL → OpenCV
            main_array = np.array(main_image)
            template_array = np.array(template_image)

            main_gray = cv2.cvtColor(
                main_array,
                cv2.COLOR_RGB2GRAY
            )

            template_gray = cv2.cvtColor(
                template_array,
                cv2.COLOR_RGB2GRAY
            )

            # Check template size
            main_height, main_width = main_gray.shape
            template_height, template_width = template_gray.shape

            if (
                template_height > main_height
                or template_width > main_width
            ):
                st.error(
                    "❌ Template image is larger than the main image."
                )
                return

            # ---------------------------------------------
            # Template Matching
            # ---------------------------------------------

            result = cv2.matchTemplate(
                main_gray,
                template_gray,
                cv2.TM_CCOEFF_NORMED
            )

            min_value, max_value, min_location, max_location = cv2.minMaxLoc(
                result
            )

            similarity = max_value * 100

            # Best match location
            x, y = max_location

            # Draw rectangle
            output_image = main_array.copy()

            cv2.rectangle(
                output_image,
                (x, y),
                (
                    x + template_width,
                    y + template_height
                ),
                (0, 255, 0),
                4
            )

            # Add label
            cv2.putText(
                output_image,
                "BEST MATCH",
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ---------------------------------------------
            # Result
            # ---------------------------------------------

            st.markdown("## 📊 ANALYSIS RESULT")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric(
                    "🎯 Similarity",
                    f"{similarity:.2f}%"
                )

            with metric2:
                st.metric(
                    "📍 X Position",
                    x
                )

            with metric3:
                st.metric(
                    "📍 Y Position",
                    y
                )

            st.divider()

            # ---------------------------------------------
            # Match status
            # ---------------------------------------------

            if similarity >= 70:

                st.success(
                    f"🟢 MATCH FOUND — Similarity {similarity:.2f}%"
                )

            else:

                st.warning(
                    f"🟡 LOW MATCH — Similarity {similarity:.2f}%"
                )

            # ---------------------------------------------
            # Result image
            # ---------------------------------------------

            st.markdown("### 🛰️ Object Hunt Result")

            st.image(
                output_image,
                caption="Best matching region highlighted",
                use_container_width=True
            )

            # ---------------------------------------------
            # Technical information
            # ---------------------------------------------

            with st.expander("🔬 View Technical Details"):

                st.write(
                    "Algorithm: OpenCV Template Matching"
                )

                st.write(
                    "Matching Method: TM_CCOEFF_NORMED"
                )

                st.write(
                    f"Template Size: "
                    f"{template_width} × {template_height} pixels"
                )

                st.write(
                    f"Detected Location: "
                    f"({x}, {y})"
                )

                st.write(
                    f"Similarity Score: "
                    f"{similarity:.2f}%"
                )

    else:

        st.info(
            "👆 Upload both the main image and target template "
            "to start the Object Hunt."
        )


