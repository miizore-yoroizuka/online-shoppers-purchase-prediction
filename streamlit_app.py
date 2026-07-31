import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Purchase Intent Scorer", layout="wide")

# ---------------------------------------------------------------- load artefacts
@st.cache_resource
def load_artefacts():
    """Load the trained model, the scaler fitted on the training set, and the
    exact column order the model expects. Cached so this runs once per session."""
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

try:
    model, scaler, feature_columns = load_artefacts()
except FileNotFoundError:
    st.error("Model files not found. model.pkl, scaler.pkl and columns.pkl must "
             "sit in the same folder as this app.")
    st.stop()

# ---------------------------------------------------------------- header
st.title("Purchase Intent Scorer")
st.markdown(
    "Score a live shopping session and decide whether to show a retention prompt "
    "before the visitor leaves. Enter what the site has observed so far."
)
st.divider()

# ---------------------------------------------------------------- inputs
left, middle, right = st.columns(3)

with left:
    st.subheader("Pages viewed")
    product_pages = st.slider("Product pages", 0, 200, 20,
                              help="How many product pages this visitor has opened.")
    product_time = st.slider("Time on product pages (seconds)", 0, 5000, 800)
    admin_pages = st.slider("Account pages", 0, 30, 2,
                            help="Login, account settings, address book.")
    admin_time = st.slider("Time on account pages (seconds)", 0, 1000, 60)
    info_pages = st.slider("Info pages", 0, 25, 0,
                           help="About us, shipping info, FAQs.")
    info_time = st.slider("Time on info pages (seconds)", 0, 1000, 0)

with middle:
    st.subheader("Engagement")
    exit_rate = st.slider("Exit rate", 0.0, 0.2, 0.03, step=0.005,
                          help="Share of this visitor's page views that were the "
                               "last page of their session. Lower is better.")
    bounce_rate = st.slider("Bounce rate", 0.0, 0.2, 0.01, step=0.005,
                            help="Share of visits that left after a single page.")
    page_value = st.slider("Page value", 0.0, 200.0, 5.0, step=0.5,
                           help="Average value of the pages viewed, from your "
                                "analytics. Zero means the visitor has not "
                                "reached checkout pages.")
    special_day = st.select_slider(
        "Closeness to a special day", options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.0,
        help="0 = ordinary day, 1 = the day itself (e.g. Valentine's Day).")

with right:
    st.subheader("Session details")
    month = st.selectbox("Month",
                         ["Aug", "Dec", "Feb", "Jul", "June", "Mar", "May",
                          "Nov", "Oct", "Sep"], index=7)
    visitor_type = st.selectbox("Visitor type",
                                ["New_Visitor", "Returning_Visitor", "Other"], index=1)
    weekend = st.checkbox("Weekend session", value=False)
    traffic_type = st.selectbox("Traffic source",
                                ["1", "2", "3", "4", "5", "6", "8", "10", "11",
                                 "13", "20", "Other"], index=1)
    region = st.selectbox("Region", [str(i) for i in range(1, 10)], index=0)
    operating_system = st.selectbox("Operating system",
                                    [str(i) for i in range(1, 9)], index=1)
    browser = st.selectbox("Browser", [str(i) for i in range(1, 14)], index=1)

st.caption(
    "Traffic source, region, operating system and browser are the numeric "
    "identifiers used by the site's analytics platform."
)

st.divider()

# ---------------------------------------------------------------- validation
def validate(product_pages, product_time, admin_pages, admin_time,
             info_pages, info_time):
    """Catch input combinations that cannot occur in a real session."""
    problems = []
    if product_pages == 0 and product_time > 0:
        problems.append("Time recorded on product pages, but no product pages viewed.")
    if admin_pages == 0 and admin_time > 0:
        problems.append("Time recorded on account pages, but no account pages viewed.")
    if info_pages == 0 and info_time > 0:
        problems.append("Time recorded on info pages, but no info pages viewed.")
    if product_pages + admin_pages + info_pages == 0:
        problems.append("No pages viewed at all. There is no session to score.")
    if bounce_rate > exit_rate:
        problems.append("Bounce rate cannot exceed exit rate: a bounce is a "
                        "special case of an exit.")
    return problems

# ---------------------------------------------------------------- build the row
def build_row():
    """Rebuild the exact 61-column feature row the model was trained on.
    Starting from zeros means every dropped reference level (traffic type 1,
    OS 1, browser 1, region 1, August, new visitor) is encoded correctly."""
    row = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    row.loc[0, "Administrative"] = admin_pages
    row.loc[0, "Administrative_Duration"] = admin_time
    row.loc[0, "Informational"] = info_pages
    row.loc[0, "Informational_Duration"] = info_time
    row.loc[0, "ProductRelated"] = product_pages
    row.loc[0, "ProductRelated_Duration"] = product_time
    row.loc[0, "BounceRates"] = bounce_rate
    row.loc[0, "ExitRates"] = exit_rate
    row.loc[0, "PageValues"] = page_value
    row.loc[0, "SpecialDay"] = special_day
    row.loc[0, "Weekend"] = 1.0 if weekend else 0.0

    # Same derivation as section 3.2 of the notebook: zero pages gives zero,
    # not a division error.
    row.loc[0, "avg_time_per_product_page"] = (
        product_time / product_pages if product_pages > 0 else 0.0
    )

    # Dummies. Reference levels are left at zero.
    for name, value in [
        (f"Month_{month}", month != "Aug"),
        (f"VisitorType_{visitor_type}", visitor_type != "New_Visitor"),
        (f"TrafficType_grouped_{traffic_type}", traffic_type != "1"),
        (f"Region_{region}", region != "1"),
        (f"OperatingSystems_{operating_system}", operating_system != "1"),
        (f"Browser_{browser}", browser != "1"),
    ]:
        if value and name in row.columns:
            row.loc[0, name] = 1.0

    return row

# ---------------------------------------------------------------- predict
if st.button("Score this session", type="primary", use_container_width=True):
    problems = validate(product_pages, product_time, admin_pages,
                        admin_time, info_pages, info_time)

    if problems:
        st.error("Please correct the following before scoring:")
        for p in problems:
            st.write(f"- {p}")
    else:
        try:
            row = build_row()
            if list(row.columns) != list(feature_columns):
                st.error("Feature mismatch. The app and the model disagree on "
                         "the input layout.")
                st.stop()

            probability = model.predict_proba(scaler.transform(row))[0][1]

            # Three bands rather than the model's own 0.5 cutoff. The middle band
            # is the one worth intervening on: high scorers convert unaided, low
            # scorers will not be persuaded, and a session near the middle is
            # where a prompt can actually change the outcome.
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Likelihood of purchase", f"{probability:.0%}")
            with col_b:
                if probability >= 0.70:
                    st.success("**Likely to buy.** No intervention needed, this "
                               "visitor is already on track to convert.")
                elif probability >= 0.30:
                    st.warning("**On the fence.** Worth a nudge: a free-delivery "
                               "reminder or a live chat prompt.")
                else:
                    st.info("**Unlikely to buy.** Browsing without strong intent. "
                            "A prompt here is unlikely to change the outcome.")

            st.progress(float(probability))
            st.caption(
                "The model is tuned to catch buyers rather than to be cautious, "
                "so it flags roughly one in four sessions. That is deliberate: "
                "a missed buyer is a lost sale, an unnecessary prompt is ignored."
            )
        except Exception as e:
            st.error(f"Something went wrong while scoring: {e}")

st.divider()
st.caption("Built on 12,330 real e-commerce sessions (UCI Online Shoppers "
           "Purchasing Intention Dataset). Model: logistic regression, tuned.")