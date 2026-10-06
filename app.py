
import streamlit as st

from MODULES.template_matching import template_matching_page
from MODULES.viola_jones import viola_jones_page
from MODULES.deepface_analysis import deepface_analysis_page
from MODULES.facenet_analysis import facenet_analysis_page


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="NEXUS VISION",
    page_icon="🧿",
    layout="wide"
)


# ==================================================
# SESSION STATE
# ==================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# ==================================================
# NEXUS VISION DESIGN
# ==================================================

st.markdown(
    """
    <style>

    /* MAIN BACKGROUND */
    .stApp {
        background:
        linear-gradient(
            135deg,
            #030712 0%,
            #081126 50%,
            #101a38 100%
        );
    }

    /* HEADINGS */
    h1 {
        color: #ffffff !important;
        font-weight: 900 !important;
    }

    h2 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* NORMAL TEXT */
    p {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* CAPTIONS */
    .stCaption {
        color: #00f5ff !important;
        font-weight: 600 !important;
    }

    /* MARKDOWN */
    .stMarkdown {
        color: #ffffff !important;
    }

    .stMarkdown p {
        color: #ffffff !important;
    }

    /* FILE UPLOADER */
    label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* METRICS */
    [data-testid="stMetricValue"] {
        color: #00f5ff !important;
        font-weight: 900 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* ALERTS */
    .stAlert {
        color: #ffffff !important;
    }

    /* BUTTONS */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 12px 18px;
        font-weight: 800;
        color: #ffffff;
        background:
        linear-gradient(
            90deg,
            #111c3a,
            #172554
        );
        border: 1px solid #00f5ff;
        transition: 0.3s;
    }

    .stButton > button:hover {
        border: 1px solid #ffffff;
        transform: translateY(-2px);
        background:
        linear-gradient(
            90deg,
            #172554,
            #1e3a8a
        );
    }

    /* DIVIDERS */
    hr {
        border-color: #24345f !important;
    }

    /* INPUT TEXT */
    input {
        color: #ffffff !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HOME DASHBOARD
# ==================================================

if st.session_state.page == "home":

    st.title("🧿 NEXUS VISION")

    st.subheader(
        "Where Vision Meets Intelligence"
    )

    st.caption(
        "AI • COMPUTER VISION • FACE ANALYTICS"
    )

    st.success(
        "🟢 VISION ENGINE ONLINE • SYSTEM READY"
    )

    st.divider()


    # ==================================================
    # SYSTEM METRICS
    # ==================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "VISION MODULES",
            "04"
        )

    with col2:
        st.metric(
            "AI ENGINE",
            "ACTIVE"
        )

    with col3:
        st.metric(
            "CV TECHNOLOGY",
            "READY"
        )

    with col4:
        st.metric(
            "SYSTEM STATUS",
            "ONLINE"
        )


    st.divider()


    # ==================================================
    # MODULE SECTION
    # ==================================================

    st.subheader(
        "🚀 Vision Intelligence Modules"
    )

    st.write(
        "Choose a module to explore the computer "
        "vision technology."
    )

    st.write("")


    # ==================================================
    # ROW 1
    # ==================================================

    col1, col2 = st.columns(2)


    # TEMPLATE MATCHING
    with col1:

        st.markdown(
            "### 🔍 OBJECT HUNT"
        )

        st.write(
            "Template Matching"
        )

        st.caption(
            "Find a specific object or pattern "
            "inside an image."
        )

        if st.button(
            "START OBJECT HUNT  →",
            key="template_button"
        ):

            st.session_state.page = "template"
            st.rerun()


    # VIOLA JONES
    with col2:

        st.markdown(
            "### 👤 FACE SCANNER"
        )

        st.write(
            "Viola–Jones Algorithm"
        )

        st.caption(
            "Detect faces using Haar Cascade "
            "face detection."
        )

        if st.button(
            "START FACE SCANNER  →",
            key="viola_button"
        ):

            st.session_state.page = "viola"
            st.rerun()


    st.write("")


    # ==================================================
    # ROW 2
    # ==================================================

    col1, col2 = st.columns(2)


    # DEEPFACE
    with col1:

        st.markdown(
            "### 🧠 AI FACE INSIGHT"
        )

        st.write(
            "DeepFace"
        )

        st.caption(
            "Analyze facial emotion using "
            "deep learning."
        )

        if st.button(
            "START AI ANALYSIS  →",
            key="deepface_button"
        ):

            st.session_state.page = "deepface"
            st.rerun()


    # FACENET
    with col2:

        st.markdown(
            "### 📐 FACE SIMILARITY LAB"
        )

        st.write(
            "FaceNet"
        )

        st.caption(
            "Compare facial features using "
            "deep learning embeddings."
        )

        if st.button(
            "START FACE COMPARISON  →",
            key="facenet_button"
        ):

            st.session_state.page = "facenet"
            st.rerun()


    st.divider()


    # ==================================================
    # FOOTER
    # ==================================================

    st.caption(
        "NEXUS VISION  •  COMPUTER VISION LAB"
    )


# ==================================================
# TEMPLATE MATCHING PAGE
# ==================================================

elif st.session_state.page == "template":

    if st.button(
        "← BACK TO NEXUS VISION",
        key="back_template"
    ):

        st.session_state.page = "home"
        st.rerun()

    st.divider()

    template_matching_page()


# ==================================================
# VIOLA–JONES PAGE
# ==================================================

elif st.session_state.page == "viola":

    if st.button(
        "← BACK TO NEXUS VISION",
        key="back_viola"
    ):

        st.session_state.page = "home"
        st.rerun()

    st.divider()

    viola_jones_page()


# ==================================================
# DEEPFACE PAGE
# ==================================================

elif st.session_state.page == "deepface":

    if st.button(
        "← BACK TO NEXUS VISION",
        key="back_deepface"
    ):

        st.session_state.page = "home"
        st.rerun()

    st.divider()

    deepface_analysis_page()


# ==================================================
# FACENET PAGE
# ==================================================

elif st.session_state.page == "facenet":

    if st.button(
        "← BACK TO NEXUS VISION",
        key="back_facenet"
    ):

        st.session_state.page = "home"
        st.rerun()

    st.divider()

    facenet_analysis_page()

