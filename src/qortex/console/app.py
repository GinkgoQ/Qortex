"""Qortex Streamlit dashboard entrypoint.

Run with ``qortex dashboard``, ``python -m qortex.console.app``, or
``streamlit run src/qortex/console/app.py``.
"""

from __future__ import annotations

import os
import shutil


def _load_streamlit():
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Qortex dashboard requires Streamlit: pip install qortex[dashboard]"
        ) from exc
    return st


def main() -> None:
    """Launch the Streamlit dashboard."""
    st = _load_streamlit()

    st.set_page_config(
        page_title="Qortex | GinkgoQ",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Qortex — OpenNeuro ML Data Lake")
    st.caption("Qortex by GinkgoQ · ML-ready neurodata from OpenNeuro")

    with st.sidebar:
        st.header("Settings")
        api_token = st.text_input("API Token (optional)", type="password")
        if api_token:
            os.environ["QORTEX_API_TOKEN"] = api_token

        st.divider()
        mode = st.selectbox("View", ["Catalog Search", "Dataset EDA", "Cache Info"])

    if mode == "Catalog Search":
        _render_catalog_search(st)
    elif mode == "Dataset EDA":
        _render_dataset_eda(st)
    elif mode == "Cache Info":
        _render_cache_info(st)


def _render_catalog_search(st) -> None:
    st.header("Catalog Search")

    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        query = st.text_input("Search datasets", placeholder="auditory, memory, ...")
    with col2:
        modality = st.selectbox("Modality", ["", "eeg", "mri", "fmri", "meg", "ieeg", "fnirs", "pet", "dwi"])
    with col3:
        min_subs = st.number_input("Min subjects", min_value=0, value=0, step=1)
    col4, col5, col6 = st.columns([2, 2, 1])
    with col4:
        task = st.text_input("Task", placeholder="rest, nback, ...")
    with col5:
        author = st.text_input("Author", placeholder="optional")
    with col6:
        has_events = st.checkbox("Events only", value=False)

    if st.button("Search", type="primary"):
        try:
            from qortex.catalog.search import search

            results = search(
                query=query or None,
                modality=modality or None,
                task=task or None,
                author=author or None,
                min_subjects=int(min_subs) if min_subs > 0 else None,
                has_events=True if has_events else None,
                limit=50,
            )
            if results:
                import polars as pl

                df = pl.DataFrame(results)
                st.dataframe(df.to_pandas(), use_container_width=True)
                st.caption(f"{len(results)} datasets found")
            else:
                st.info("No results. Try refreshing the catalog first.")
        except Exception as exc:
            st.error(f"Error: {exc}")

    if st.button("Refresh Catalog from OpenNeuro"):
        with st.spinner("Fetching from OpenNeuro ..."):
            try:
                from qortex.catalog.refresh import refresh

                n = refresh(progress=False)
                st.success(f"{n} datasets indexed.")
            except Exception as exc:
                st.error(f"Refresh failed: {exc}")


def _render_dataset_eda(st) -> None:
    st.header("Dataset EDA")

    dataset_id = st.text_input("Dataset ID", placeholder="ds004130")
    snapshot_tag = st.text_input("Snapshot (leave blank for latest)", placeholder="1.0.0")

    if st.button("Run EDA", type="primary") and dataset_id:
        with st.spinner(f"Fetching manifest for {dataset_id} ..."):
            try:
                from qortex.client.graphql import OpenNeuroClient
                from qortex.eda.report import EDAEngine
                from qortex.manifest.builder import ManifestBuilder

                client = OpenNeuroClient()
                builder = ManifestBuilder()
                snap = snapshot_tag.strip() or None
                snap_ref = client.get_snapshot(dataset_id, snap) if snap else client.get_latest_snapshot(dataset_id)
                raw_files = client.get_files(dataset_id, snap_ref.tag)
                manifest = builder.build(dataset_id, snap_ref, raw_files)

                engine = EDAEngine(manifest)
                report = engine.run()

            except Exception as exc:
                st.error(f"Failed: {exc}")
                st.stop()

        q = report.quality
        c1, c2, c3 = st.columns(3)
        c1.metric("BIDS Score", f"{q.bids_score:.0f}/100")
        c2.metric("ML-Readiness", f"{q.ml_readiness_score:.0f}/100")
        c3.metric("Loadability", f"{q.loadability_score:.0f}/100")

        s = report.summary
        if s:
            st.subheader("Overview")
            st.json({
                "files": s.n_files,
                "subjects": s.n_subjects,
                "sessions": s.n_sessions,
                "tasks": s.n_tasks,
                "size_GB": round(s.total_size / 1e9, 2),
                "modalities": s.modalities,
            })

        for _name, fig in report.figures.items():
            try:
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not render a figure: {exc}")

        if q.issues:
            st.subheader("Issues")
            for issue in q.issues:
                st.warning(issue)
        if q.risks:
            st.subheader("ML Risks")
            for risk in q.risks:
                st.warning(risk)

        if report.html:
            st.download_button(
                "Download HTML Report",
                data=report.html,
                file_name=f"{dataset_id}_eda.html",
                mime="text/html",
            )


def _render_cache_info(st) -> None:
    st.header("Cache Info")

    try:
        from qortex.core.config import get_config

        cfg = get_config()
        cache_dir = cfg.cache_dir
        if cache_dir.exists():
            total = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            st.metric("Cache directory", str(cache_dir))
            st.metric("Total size", f"{total / 1e9:.2f} GB")

            datasets_dir = cache_dir / "datasets"
            if datasets_dir.exists():
                dsets = [d.name for d in datasets_dir.iterdir() if d.is_dir()]
                st.write(f"**Downloaded datasets ({len(dsets)}):**")
                for dataset in sorted(dsets):
                    st.text(f"  - {dataset}")
        else:
            st.info("Cache directory does not exist yet.")
    except Exception as exc:
        st.error(f"Error: {exc}")

    if st.button("Clear Cache", type="secondary") and st.checkbox("I confirm I want to delete the cache"):
        from qortex.core.config import get_config

        shutil.rmtree(get_config().cache_dir, ignore_errors=True)
        st.success("Cache cleared.")


if __name__ == "__main__":
    main()
