"""Streamlit dashboard for SupportPilot AI."""

import altair as alt
import pandas as pd
import streamlit as st

from supportpilot.ui.batch import (
    run_batch_inference,
)
from supportpilot.ui.api_client import (
    SupportPilotAPIClient,
    SupportPilotAPIError,
)
from supportpilot.ui.formatting import (
    format_intent,
    format_percent,
    format_threshold,
)


st.set_page_config(
    page_title="SupportPilot AI",
    page_icon="💬",
    layout="wide",
)


@st.cache_resource
def get_api_client() -> SupportPilotAPIClient:
    """Create and reuse one API client."""

    return SupportPilotAPIClient()


client = get_api_client()


st.title("SupportPilot AI")
st.caption(
    "Customer support intent classification "
    "with confidence-aware fallback."
)


with st.sidebar:
    st.header("System Status")

    try:
        health = client.health()

        st.success("API Online")

        st.metric(
            "Model",
            "Loaded"
            if health["model_loaded"]
            else "Unavailable",
        )

        st.metric(
            "Device",
            health["device"].upper(),
        )

        st.metric(
            "Intent Labels",
            health["num_labels"],
        )

    except SupportPilotAPIError as error:
        st.error("API Offline")
        st.caption(str(error))


tab_single, tab_top_k, tab_batch = st.tabs(
    [
        "Intent Analyzer",
        "Top-K Analysis",
        "Batch Analysis",
    ]
)


with tab_single:
    st.subheader("Intent Analyzer")

    message = st.text_area(
        "Customer message",
        placeholder="Where is my order?",
        height=120,
    )

    if st.button(
        "Analyze Intent",
        type="primary",
    ):
        if not message.strip():
            st.warning(
                "Please enter a customer message."
            )

        else:
            try:
                result = client.predict(
                    message.strip()
                )

            except SupportPilotAPIError as error:
                st.error(str(error))

            else:
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Final Intent",
                    format_intent(
                        result["final_intent"]
                    ),
                )

                col2.metric(
                    "Confidence",
                    format_percent(
                        result[
                            "confidence_percent"
                        ]
                    ),
                )

                col3.metric(
                    "Margin",
                    format_percent(
                        result[
                            "confidence_margin_percent"
                        ]
                    ),
                )

                if result["accepted"]:
                    st.success(
                        "Prediction accepted."
                    )

                else:
                    st.warning(
                        "Prediction sent to fallback."
                    )

                st.divider()

                st.write(
                    "**Raw prediction:**",
                    format_intent(
                        result[
                            "predicted_intent"
                        ]
                    ),
                )

                st.write(
                    "**Second-best intent:**",
                    format_intent(
                        result[
                            "second_best_intent"
                        ]
                    ),
                )

                st.write(
                    "**Confidence threshold:**",
                    format_threshold(
                        result[
                            "min_confidence"
                        ]
                    ),
                )

                st.write(
                    "**Margin threshold:**",
                    format_threshold(
                        result[
                            "min_margin"
                        ]
                    ),
                )

                st.write(
                    "**Decision reason:**",
                    result["reason"],
                )


with tab_top_k:
    st.subheader("Top-K Analysis")

    top_k_message = st.text_area(
        "Customer message",
        placeholder="Where is my package?",
        height=120,
        key="top_k_message",
    )

    top_k_count = st.slider(
        "Number of intent candidates",
        min_value=2,
        max_value=10,
        value=5,
        key="top_k_count",
    )

    if st.button(
        "Analyze Top-K",
        type="primary",
        key="top_k_button",
    ):
        if not top_k_message.strip():
            st.warning(
                "Please enter a customer message."
            )

        else:
            try:
                result = client.predict_top_k(
                    top_k_message.strip(),
                    top_k=top_k_count,
                )

            except SupportPilotAPIError as error:
                st.error(str(error))

            else:
                predictions = result["predictions"]

                st.success(
                    f"Returned {len(predictions)} "
                    "intent candidates."
                )

                table_data = []

                for prediction in predictions:
                    table_data.append(
                        {
                            "Rank": prediction["rank"],
                            "Intent": format_intent(
                                prediction[
                                    "predicted_intent"
                                ]
                            ),
                            "Confidence (%)": prediction[
                                "confidence_percent"
                            ],
                        }
                    )

                dataframe = pd.DataFrame(
                    table_data
                )

                st.dataframe(
                    dataframe,
                    use_container_width=True,
                    hide_index=True,
                )

                chart = (
                    alt.Chart(dataframe)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "Confidence (%):Q",
                            title="Confidence (%)",
                        ),
                        y=alt.Y(
                            "Intent:N",
                            sort=dataframe["Intent"].tolist(),
                            title="Intent",
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Rank:Q",
                                title="Rank",
                            ),
                            alt.Tooltip(
                                "Intent:N",
                                title="Intent",
                            ),
                            alt.Tooltip(
                                "Confidence (%):Q",
                                title="Confidence",
                                format=".4f",
                            ),
                        ],
                    )
                )
                
                st.altair_chart(
                    chart,
                    use_container_width=True,
                )


with tab_batch:
    st.subheader("Batch Analysis")

    st.caption(
        "Analyze multiple customer messages "
        "from manual input or a CSV file."
    )

    input_mode = st.radio(
        "Input source",
        options=[
            "Manual Input",
            "CSV Upload",
        ],
        horizontal=True,
    )

    texts: list[str] = []

    if input_mode == "Manual Input":
        batch_text = st.text_area(
            "Customer messages",
            placeholder=(
                "Where is my order?\n"
                "I want to cancel my order\n"
                "What is the weather today?"
            ),
            height=180,
            help="Enter one message per line.",
        )

        texts = [
            line.strip()
            for line in batch_text.splitlines()
            if line.strip()
        ]

        if texts:
            st.caption(
                f"{len(texts)} message(s) ready."
            )

    else:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
        )

        if uploaded_file is not None:
            try:
                uploaded_dataframe = pd.read_csv(
                    uploaded_file
                )

            except Exception as error:
                st.error(
                    f"Unable to read CSV: {error}"
                )

            else:
                if uploaded_dataframe.empty:
                    st.warning(
                        "The uploaded CSV is empty."
                    )

                else:
                    st.dataframe(
                        uploaded_dataframe.head(),
                        use_container_width=True,
                        hide_index=True,
                    )

                    text_column = st.selectbox(
                        "Message column",
                        options=list(
                            uploaded_dataframe.columns
                        ),
                    )

                    texts = (
                        uploaded_dataframe[
                            text_column
                        ]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )

                    texts = [
                        text
                        for text in texts
                        if text
                    ]

                    st.caption(
                        f"{len(texts)} message(s) "
                        "loaded from CSV."
                    )

    inference_batch_size = st.slider(
        "Inference batch size",
        min_value=1,
        max_value=128,
        value=32,
    )

    if st.button(
        "Run Batch Analysis",
        type="primary",
        key="batch_button",
    ):
        if not texts:
            st.warning(
                "Please provide at least one "
                "customer message."
            )

        else:
            try:
                with st.spinner(
                    f"Analyzing {len(texts)} message(s)..."
                ):
                    result = run_batch_inference(
                        client,
                        texts,
                        batch_size=inference_batch_size,
                    )

            except SupportPilotAPIError as error:
                st.error(str(error))

            else:
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Messages",
                    result["total"],
                )

                col2.metric(
                    "Accepted",
                    result["accepted"],
                )

                col3.metric(
                    "Fallback",
                    result["fallback"],
                )

                predictions = []

                for prediction in result[
                    "predictions"
                ]:
                    predictions.append(
                        {
                            "Message": prediction[
                                "text"
                            ],
                            "Raw Intent": format_intent(
                                prediction[
                                    "predicted_intent"
                                ]
                            ),
                            "Final Intent": format_intent(
                                prediction[
                                    "final_intent"
                                ]
                            ),
                            "Confidence (%)": prediction[
                                "confidence_percent"
                            ],
                            "Margin (%)": prediction[
                                "confidence_margin_percent"
                            ],
                            "Accepted": prediction[
                                "accepted"
                            ],
                            "Reason": prediction[
                                "reason"
                            ],
                        }
                    )

                result_dataframe = pd.DataFrame(
                    predictions
                )

                st.divider()

                st.subheader("Prediction Results")

                st.dataframe(
                    result_dataframe,
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader(
                    "Final Intent Distribution"
                )

                distribution = (
                    result_dataframe[
                        "Final Intent"
                    ]
                    .value_counts()
                    .rename_axis("Intent")
                    .reset_index(name="Count")
                )

                st.bar_chart(
                    distribution,
                    x="Intent",
                    y="Count",
                )

                csv_data = (
                    result_dataframe
                    .to_csv(
                        index=False
                    )
                    .encode("utf-8")
                )

                st.download_button(
                    "Download Results as CSV",
                    data=csv_data,
                    file_name=(
                        "supportpilot_predictions.csv"
                    ),
                    mime="text/csv",
                )