"""Monitoring dashboard: API health, data drift, and model/retraining history.

Reads straight from the Postgres tables the API writes (requests, feedback)
and the retraining pipeline writes (retraining_runs), plus /health on the API
for the model currently being served.

    streamlit run milestone_4/dashboard/app.py
"""

import json
import os
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import altair as alt
import drift
import httpx
import pandas as pd
import psycopg
import streamlit as st

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://app:app@localhost:5432/tweets")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
REFERENCE_PATH = Path(__file__).resolve().parent / "reference_stats.json"
DRIFT_SAMPLE = 2000  # most recent successful requests compared against training data

DISPLAY_TZ = os.environ.get("DISPLAY_TZ", "Asia/Bangkok")

# window -> (lookback, time bucket)
WINDOWS = {
    "Last 15 minutes": (timedelta(minutes=15), timedelta(seconds=15)),
    "Last hour": (timedelta(hours=1), timedelta(minutes=1)),
    "Last 6 hours": (timedelta(hours=6), timedelta(minutes=5)),
    "Last 24 hours": (timedelta(hours=24), timedelta(minutes=15)),
    "Last 7 days": (timedelta(days=7), timedelta(hours=1)),
}

# reference palette (dataviz skill): categorical slots 1-2, neutral for "reference"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9893"
LEVEL_BADGE = {"stable": "✅ stable", "moderate": "⚠️ moderate", "major": "🛑 major"}

st.set_page_config(page_title="Disaster Tweet Monitor", page_icon="📡", layout="wide")


@st.cache_resource
def load_reference() -> dict:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
        cur = conn.execute(sql, params)
        columns = [c.name for c in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=columns)
    # Postgres avg() returns NUMERIC -> Decimal, which the chart frontend
    # can't serialize; the dashboard only needs floats
    for column in df.columns:
        if df[column].map(lambda v: isinstance(v, Decimal)).any():
            df[column] = df[column].astype(float)
    return df


def backend_health() -> dict | None:
    try:
        return httpx.get(f"{BACKEND_URL}/health", timeout=2).json()
    except (httpx.HTTPError, ValueError):
        return None


def local_times(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty:
        return df
    stamps = pd.to_datetime(df[column], utc=True).dt.tz_convert(DISPLAY_TZ)
    return df.assign(**{column: stamps.dt.strftime("%Y-%m-%d %H:%M:%S")})


def line_chart(
    df: pd.DataFrame,
    y: str,
    title: str,
    y_title: str,
    color: str = BLUE,
    series: str | None = None,
    y_format: str = ",.0f",
    zero: bool = True,
) -> alt.Chart:
    df = df.dropna(subset=[y]).sort_values("bucket").copy()
    # A new line segment starts wherever buckets are missing (no traffic),
    # so idle periods show as gaps instead of a straight line through them.
    gaps = df.groupby(series)["bucket"].diff() if series else df["bucket"].diff()
    df["segment"] = (gaps > bucket * 1.5).cumsum()
    # wall-clock time in DISPLAY_TZ without a zone: naive times are drawn as
    # they are, so every viewer sees the same clock whatever their own timezone
    df["time"] = df["bucket"].dt.tz_convert(DISPLAY_TZ).dt.tz_localize(None)
    df["time_label"] = df["time"].dt.strftime("%H:%M:%S")
    base = alt.Chart(df).encode(
        x=alt.X(
            "time:T",
            title=f"time ({DISPLAY_TZ})",
            axis=alt.Axis(grid=False, format="%H:%M"),
        ),
        y=alt.Y(
            f"{y}:Q",
            title=y_title,
            scale=alt.Scale(zero=zero),
            axis=alt.Axis(format=y_format, gridOpacity=0.3),
        ),
        tooltip=[
            alt.Tooltip("time_label:N", title="Time"),
            *([alt.Tooltip(f"{series}:N", title="Series")] if series else []),
            alt.Tooltip(f"{y}:Q", title=y_title, format=y_format),
        ],
    )
    if series:
        palette = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4"]
        encoded = base.encode(
            color=alt.Color(
                f"{series}:N",
                title=None,
                scale=alt.Scale(range=palette),
                legend=alt.Legend(orient="top"),
            )
        )
    else:
        encoded = base.encode(color=alt.value(color))
    lines = encoded.encode(detail="segment:N").mark_line(strokeWidth=2)
    # invisible points with a large hit area carry the hover tooltip
    points = encoded.mark_point(filled=True, size=150, opacity=0)
    return (lines + points).properties(title=title, height=220)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📡 Monitor")
window_label = st.sidebar.selectbox("Time window", list(WINDOWS), index=1)
auto_refresh = st.sidebar.toggle("Auto-refresh every 10 s", value=True)
lookback, bucket = WINDOWS[window_label]


@st.fragment(run_every="10s" if auto_refresh else None)
def render() -> None:
    health = backend_health()
    try:
        ops = query(
            """
            SELECT date_bin(%s::interval, ts, TIMESTAMPTZ '2000-01-01') AS bucket,
                   count(*) AS requests,
                   count(*) FILTER (WHERE status_code >= 400) AS errors,
                   count(*) FILTER (WHERE status_code >= 500) AS server_errors,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)
                       FILTER (WHERE status_code = 200) AS p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
                       FILTER (WHERE status_code = 200) AS p95,
                   avg(confidence) FILTER (WHERE status_code = 200 AND NOT cached)
                       AS confidence,
                   avg(cached::int) FILTER (WHERE status_code = 200) AS cache_hit_rate
            FROM requests
            WHERE ts > now() - %s::interval
            GROUP BY bucket ORDER BY bucket
            """,
            (bucket, lookback),
        )
        totals = query(
            """
            SELECT count(*) AS requests,
                   avg((status_code >= 400)::int) AS error_rate,
                   avg((status_code >= 500)::int) AS server_error_rate,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)
                       FILTER (WHERE status_code = 200) AS p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
                       FILTER (WHERE status_code = 200) AS p95,
                   avg(confidence) FILTER (WHERE status_code = 200 AND NOT cached)
                       AS confidence,
                   avg(cached::int) FILTER (WHERE status_code = 200) AS cache_hit_rate
            FROM requests WHERE ts > now() - %s::interval
            """,
            (lookback,),
        ).iloc[0]
    except psycopg.OperationalError as exc:
        st.error(f"Can't reach the database at {DATABASE_URL}: {exc}")
        return

    # ------------------------------------------------------------ KPI row
    st.title("Disaster Tweet Classifier: Monitoring")
    serving = health["model_version"] if health else "API unreachable"
    status = health["status"] if health else "down"
    st.caption(f"{window_label} · serving model **{serving}** · API status **{status}**")

    def pct(x) -> str:
        return "–" if pd.isna(x) else f"{x * 100:.1f}%"

    cols = st.columns(6)
    cols[0].metric("Requests", f"{int(totals.requests):,}")
    cols[1].metric("Error rate (4xx+5xx)", pct(totals.error_rate))
    cols[2].metric("Server errors (5xx)", pct(totals.server_error_rate))
    latency = "–" if pd.isna(totals.p50) else f"{totals.p50:,.0f} / {totals.p95:,.0f} ms"
    cols[3].metric("Latency p50 / p95", latency)
    cols[4].metric(
        "Mean confidence", "–" if pd.isna(totals.confidence) else f"{totals.confidence:.3f}"
    )
    cols[5].metric("Cache hit rate", pct(totals.cache_hit_rate))

    ops_tab, drift_tab, model_tab = st.tabs(["Operations", "Data drift", "Model & retraining"])
    with ops_tab:
        render_operations(ops)
    with drift_tab:
        render_drift(lookback)
    with model_tab:
        render_models(lookback, bucket)


def render_operations(ops: pd.DataFrame) -> None:
    if ops.empty:
        st.info("No requests in this window yet. Start the workload generator.")
        return
    seconds = bucket.total_seconds()
    ops = ops.assign(
        throughput=ops.requests / seconds,
        error_rate=ops.errors / ops.requests,
    )
    latency = ops.melt(
        id_vars="bucket", value_vars=["p50", "p95"], var_name="percentile", value_name="latency_ms"
    )

    left, right = st.columns(2)
    left.altair_chart(
        line_chart(ops, "throughput", "Throughput", "requests / s", y_format=",.1f"),
        width="stretch",
    )
    right.altair_chart(
        line_chart(
            latency, "latency_ms", "Latency (successful requests)", "ms", series="percentile"
        ),
        width="stretch",
    )
    left, right = st.columns(2)
    left.altair_chart(
        line_chart(
            ops, "error_rate", "Error rate", "share of requests", color=ORANGE, y_format=".0%"
        ),
        width="stretch",
    )
    right.altair_chart(
        line_chart(
            ops,
            "confidence",
            "Mean prediction confidence (uncached)",
            "confidence",
            y_format=".2f",
            zero=False,
        ),
        width="stretch",
    )

    with st.expander("Table view"):
        st.dataframe(ops, hide_index=True, width="stretch")


def render_drift(lookback: timedelta) -> None:
    reference = load_reference()
    live = query(
        """
        SELECT text, prediction FROM requests
        WHERE status_code = 200 AND ts > now() - %s::interval
        ORDER BY ts DESC LIMIT %s
        """,
        (lookback, DRIFT_SAMPLE),
    )
    if len(live) < 50:
        st.info(
            f"Need at least 50 successful requests in the window to measure drift "
            f"(have {len(live)})."
        )
        return

    report = drift.compare(reference, live.text.tolist(), live.prediction.tolist())
    worst = max(report["numeric"].values(), key=lambda f: f["psi"])
    oov_shift = report["oov_rate"] - report["reference_oov_rate"]
    mix_shift = report["positive_rate"] - report["reference_positive_rate"]

    st.caption(
        f"Latest {report['num_texts']:,} successful requests vs. "
        f"{reference['num_texts']:,} training tweets (train.csv). "
        "PSI: under 0.1 stable, 0.1 to 0.25 moderate, over 0.25 major."
    )
    cols = st.columns(3)
    cols[0].metric(
        "Worst feature PSI", f"{worst['psi']:.3f}", LEVEL_BADGE[worst["level"]], delta_color="off"
    )
    cols[1].metric(
        "Out-of-vocabulary words",
        f"{report['oov_rate']:.1%}",
        f"{oov_shift:+.1%} vs training",
        delta_color="inverse",
    )
    cols[2].metric(
        "Predicted 'disaster'",
        f"{report['positive_rate']:.1%}",
        f"{mix_shift:+.1%} vs training labels",
        delta_color="off",
    )

    rows = [
        {
            "Feature": drift.NUMERIC_FEATURES[name],
            "PSI": round(f["psi"], 3),
            "Status": LEVEL_BADGE[f["level"]],
            "Training mean": round(f["reference_mean"], 3),
            "Live mean": round(f["live_mean"], 3),
        }
        for name, f in report["numeric"].items()
    ]
    left, right = st.columns([3, 2])
    left.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with right:
        st.markdown("**Most frequent words not in the training vocabulary**")
        new_words = pd.DataFrame(report["top_new_words"], columns=["word", "count"])
        st.dataframe(new_words, hide_index=True, width="stretch", height=250)

    name = st.selectbox(
        "Compare distributions",
        list(drift.NUMERIC_FEATURES),
        format_func=drift.NUMERIC_FEATURES.get,
    )
    f = report["numeric"][name]
    edges = f["edges"]
    labels = (
        [f"≤ {edges[0]:g}"]
        + [f"{lo:g} to {hi:g}" for lo, hi in zip(edges, edges[1:], strict=False)]
        + [f"> {edges[-1]:g}"]
    )
    hist = pd.DataFrame(
        {
            "bucket": labels * 2,
            "order": list(range(len(labels))) * 2,
            "share": f["reference_hist"] + f["live_hist"],
            "source": ["Training"] * len(labels) + ["Live"] * len(labels),
        }
    )
    chart = (
        alt.Chart(hist)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "bucket:N",
                sort=alt.SortField("order"),
                title=drift.NUMERIC_FEATURES[name],
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset("source:N", sort=["Training", "Live"]),
            y=alt.Y(
                "share:Q", title="share of tweets", axis=alt.Axis(format=".0%", gridOpacity=0.3)
            ),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(domain=["Training", "Live"], range=[GRAY, BLUE]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["source", "bucket", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")
    render_drift_timeline(reference, lookback)


def render_drift_timeline(reference: dict, lookback: timedelta) -> None:
    """The cards above describe only the latest requests; this shows when
    drift happened across the whole window."""
    rows = query(
        """
        SELECT date_bin(%s::interval, ts, TIMESTAMPTZ '2000-01-01') AS bucket,
               text, prediction
        FROM requests
        WHERE status_code = 200 AND ts > now() - %s::interval
        ORDER BY ts DESC LIMIT 50000
        """,
        (bucket, lookback),
    )
    points = []
    for when, group in rows.groupby("bucket"):
        if len(group) < 30:
            continue
        report = drift.compare(reference, group.text.tolist(), group.prediction.tolist())
        points.append(
            {
                "bucket": when,
                "oov_rate": report["oov_rate"],
                "worst_psi": max(f["psi"] for f in report["numeric"].values()),
            }
        )
    if not points:
        return
    timeline = pd.DataFrame(points)
    st.markdown("**Drift over the selected window**")
    left, right = st.columns(2)
    oov = line_chart(
        timeline, "oov_rate", "Out-of-vocabulary word rate", "share of words", y_format=".0%"
    )
    baseline = (
        alt.Chart(pd.DataFrame({"y": [reference["oov_rate"]]}))
        .mark_rule(color=GRAY, strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    left.altair_chart(oov + baseline, width="stretch")
    psi_chart = line_chart(
        timeline, "worst_psi", "Worst feature PSI", "PSI", color=ORANGE, y_format=".2f"
    )
    thresholds = (
        alt.Chart(pd.DataFrame({"y": [drift.PSI_MODERATE, drift.PSI_MAJOR]}))
        .mark_rule(color=GRAY, strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    right.altair_chart(psi_chart + thresholds, width="stretch")
    st.caption(
        "Dashed lines: the training set's own out-of-vocabulary rate (left); "
        "PSI thresholds 0.1 and 0.25 (right)."
    )


def render_models(lookback: timedelta, bucket: timedelta) -> None:
    per_version = query(
        """
        SELECT r.model_version AS "Model",
               count(*) AS "Requests",
               min(r.ts) AS "First seen",
               avg(r.confidence) FILTER (WHERE NOT r.cached) AS "Mean confidence",
               count(f.request_id) AS "Labelled",
               avg((r.prediction = f.true_label)::int) AS "Labelled accuracy"
        FROM requests r LEFT JOIN feedback f ON f.request_id = r.id
        WHERE r.status_code = 200 AND r.ts > now() - %s::interval
        GROUP BY r.model_version ORDER BY min(r.ts)
        """,
        (lookback,),
    )
    st.markdown("**Served models in this window**")
    st.dataframe(local_times(per_version, "First seen"), hide_index=True, width="stretch")

    accuracy = query(
        """
        SELECT date_bin(%s::interval, r.ts, TIMESTAMPTZ '2000-01-01') AS bucket,
               r.model_version, avg((r.prediction = f.true_label)::int) AS accuracy
        FROM requests r JOIN feedback f ON f.request_id = r.id
        WHERE r.ts > now() - %s::interval
        GROUP BY bucket, r.model_version ORDER BY bucket
        """,
        (bucket, lookback),
    )
    if not accuracy.empty:
        st.altair_chart(
            line_chart(
                accuracy,
                "accuracy",
                "Accuracy on requests whose label came back",
                "accuracy",
                series="model_version",
                y_format=".0%",
                zero=False,
            ),
            width="stretch",
        )

    runs = query(
        """
        SELECT id AS "Run", started_at AS "Started", trigger AS "Trigger",
               status AS "Status", base_version AS "From", candidate_version AS "To",
               n_new_labeled AS "New labels", n_train AS "Train", n_eval AS "Eval",
               base_f1 AS "Old F1", candidate_f1 AS "New F1",
               candidate_int8_f1 AS "New F1 (INT8)",
               base_guard_acc AS "Old guard acc", candidate_guard_acc AS "New guard acc",
               finished_at - started_at AS "Duration", message AS "Message"
        FROM retraining_runs ORDER BY started_at DESC LIMIT 50
        """
    )
    st.markdown("**Retraining runs**")
    if runs.empty:
        st.info("No retraining runs yet.")
    else:
        st.dataframe(local_times(runs, "Started"), hide_index=True, width="stretch")


render()
