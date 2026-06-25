"""Qortex Streamlit dashboard — run via `qortex dashboard` or `streamlit run`."""

from __future__ import annotations

try:
    import streamlit as st
except ImportError:
    raise ImportError(
        "Qortex dashboard requires Streamlit: pip install qortex[dashboard]"
    )

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Qortex | GinkgoQ",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Qortex — OpenNeuro ML Data Lake")
st.caption("Qortex by GinkgoQ · ML-ready neurodata from OpenNeuro")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    api_token = st.text_input("API Token (optional)", type="password")
    if api_token:
        import os
        os.environ["QORTEX_API_TOKEN"] = api_token

    st.divider()
    mode = st.selectbox("View", ["Catalog Search", "Dataset EDA", "Cache Info"])


# ── Catalog Search ────────────────────────────────────────────────────────────

if mode == "Catalog Search":
    st.header("Catalog Search")

    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        query = st.text_input("Search datasets", placeholder="auditory, memory, …")
    with col2:
        modality = st.selectbox("Modality", ["", "eeg", "mri", "fmri", "meg", "ieeg", "fnirs", "pet", "dwi"])
    with col3:
        min_subs = st.number_input("Min subjects", min_value=0, value=0, step=1)

    if st.button("Search", type="primary"):
        try:
            from qortex.catalog.search import search
            results = search(
                query=query or None,
                modality=modality or None,
                min_subjects=int(min_subs) if min_subs > 0 else None,
                limit=50,
            )
            if results:
                import polars as pl
                df = pl.DataFrame(results)
                st.dataframe(df.to_pandas(), use_container_width=True)
                st.caption(f"{len(results)} datasets found")
            else:
                st.info("No results. Try refreshing the catalog first.")
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("Refresh Catalog from OpenNeuro"):
        with st.spinner("Fetching from OpenNeuro …"):
            try:
                from qortex.catalog.refresh import refresh
                n = refresh(progress=False)
                st.success(f"{n} datasets indexed.")
            except Exception as e:
                st.error(f"Refresh failed: {e}")


# ── Dataset EDA ───────────────────────────────────────────────────────────────

elif mode == "Dataset EDA":
    st.header("Dataset EDA")

    dataset_id = st.text_input("Dataset ID", placeholder="ds004130")
    snapshot_tag = st.text_input("Snapshot (leave blank for latest)", placeholder="1.0.0")

    if st.button("Run EDA", type="primary") and dataset_id:
        with st.spinner(f"Fetching manifest for {dataset_id} …"):
            try:
                from qortex.client.graphql import OpenNeuroClient
                from qortex.eda.report import EDAEngine
                from qortex.manifest.builder import ManifestBuilder

                client = OpenNeuroClient()
                builder = ManifestBuilder()
                snap = snapshot_tag.strip() or None
                if snap:
                    snap_ref = client.get_snapshot(dataset_id, snap)
                else:
                    snap_ref = client.get_latest_snapshot(dataset_id)
                raw_files = client.get_files(dataset_id, snap_ref.tag)
                manifest = builder.build(dataset_id, snap_ref, raw_files)

                engine = EDAEngine(manifest)
                report = engine.run()

            except Exception as e:
                st.error(f"Failed: {e}")
                st.stop()

        # Quality scores
        q = report.quality
        c1, c2, c3 = st.columns(3)
        c1.metric("BIDS Score", f"{q.bids_score:.0f}/100")
        c2.metric("ML-Readiness", f"{q.ml_readiness_score:.0f}/100")
        c3.metric("Loadability", f"{q.loadability_score:.0f}/100")

        # Summary
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

        # Figures
        for name, fig in report.figures.items():
            try:
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        # Issues and risks
        if q.issues:
            st.subheader("Issues")
            for i in q.issues:
                st.warning(i)
        if q.risks:
            st.subheader("ML Risks")
            for r in q.risks:
                st.warning(r)

        # HTML download
        if report.html:
            st.download_button(
                "Download HTML Report",
                data=report.html,
                file_name=f"{dataset_id}_eda.html",
                mime="text/html",
            )


# ── Cache Info ────────────────────────────────────────────────────────────────

elif mode == "Cache Info":
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
                for d in sorted(dsets):
                    st.text(f"  • {d}")
        else:
            st.info("Cache directory does not exist yet.")
    except Exception as e:
        st.error(f"Error: {e}")

    if st.button("Clear Cache", type="secondary"):
        if st.checkbox("I confirm I want to delete the cache"):
            import shutil
            from qortex.core.config import get_config
            shutil.rmtree(get_config().cache_dir, ignore_errors=True)
            st.success("Cache cleared.")
