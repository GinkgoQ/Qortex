"""Selection DSL — translate user-facing filter parameters into file sets.

The selector is pure logic: it takes a Manifest and a SelectionSpec and
returns (included_files, excluded_files, warnings).  No I/O.
"""

from __future__ import annotations

from qortex._internal.glob import apply_include_exclude, find_close_matches, is_dotfile
from qortex.core.entities import FileRecord, Manifest, SelectionReason, SelectionSpec
from qortex.core.exceptions import SelectionError
from qortex.manifest.graph import ManifestGraph

# Top-level BIDS metadata files that must always be included
ESSENTIAL_FILENAMES = frozenset({
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "README",
    "CHANGES",
    ".bidsignore",
})


class Selector:
    """Resolve a ``SelectionSpec`` against a ``Manifest`` into a file list."""

    def resolve(
        self,
        manifest: Manifest,
        spec: SelectionSpec,
    ) -> tuple[list[FileRecord], list[FileRecord], list[str]]:
        files, essential, warnings, _reasons, _recordings = self.resolve_with_reasons(
            manifest, spec
        )
        return files, essential, warnings

    def resolve_with_reasons(
        self,
        manifest: Manifest,
        spec: SelectionSpec,
    ) -> tuple[
        list[FileRecord],
        list[FileRecord],
        list[str],
        dict[str, list[SelectionReason]],
        list,
    ]:
        """Return (selected_files, essential_files, warnings).

        *selected_files* includes essential files; duplicates are eliminated.
        *essential_files* is the raw list of essential files found in the manifest
        (useful for the plan to label them separately).
        """
        warnings: list[str] = []
        reasons: dict[str, list[SelectionReason]] = {}
        all_files = [f for f in manifest.files if not f.is_dir]
        all_paths = [f.path for f in all_files]
        graph = ManifestGraph(manifest)
        recordings = graph.recordings()

        def add_reason(file: FileRecord, reason: str, source: str = "selector", recording_id: str | None = None) -> None:
            reasons.setdefault(file.path, []).append(
                SelectionReason(
                    path=file.path,
                    reason=reason,
                    source=source,
                    recording_id=recording_id,
                )
            )

        # ── 1. Essential files (always included) ──────────────────────────
        essential = [f for f in all_files if f.filename in ESSENTIAL_FILENAMES]
        for file in essential:
            add_reason(file, "essential BIDS/OpenNeuro metadata")

        # ── 2. Exclude derivatives by default ─────────────────────────────
        candidates = all_files
        if not spec.include_derivatives:
            candidates = [
                f for f in candidates
                if not f.path.startswith("derivatives/")
            ]
            deriv_count = len(all_files) - len(candidates)
            if deriv_count > 0:
                warnings.append(
                    f"{deriv_count} derivative files excluded "
                    f"(use include_derivatives=True to include them)."
                )

        # ── 3. Metadata-only mode ─────────────────────────────────────────
        if spec.metadata_only:
            candidates = [
                f for f in candidates
                if f.extension in {".json", ".tsv", ".bvec", ".bval"}
                or f.filename in ESSENTIAL_FILENAMES
            ]
            for file in candidates:
                add_reason(file, "metadata-only selection")

        # ── 4. Subject filter ─────────────────────────────────────────────
        if spec.subjects:
            sub_values = {s.removeprefix("sub-") for s in spec.subjects}
            candidates = [
                f for f in candidates
                if f.entities.subject is None or f.entities.subject in sub_values
            ]

        # ── 5. Session filter ─────────────────────────────────────────────
        if spec.sessions:
            ses_values = {s.removeprefix("ses-") for s in spec.sessions}
            candidates = [
                f for f in candidates
                if f.entities.session is None or f.entities.session in ses_values
            ]

        # ── 6. Task filter ────────────────────────────────────────────────
        if spec.tasks:
            task_values = set(spec.tasks)
            candidates = [
                f for f in candidates
                if f.entities.task is None or f.entities.task in task_values
            ]

        # ── 7. Modality filter ────────────────────────────────────────────
        if spec.modalities:
            mod_values = set(spec.modalities)
            candidates = [
                f for f in candidates
                if f.modality is None or f.modality in mod_values
            ]

        # ── 8. Datatype filter ────────────────────────────────────────────
        if spec.datatypes:
            dt_values = set(spec.datatypes)
            candidates = [
                f for f in candidates
                if f.datatype is None or f.datatype in dt_values
            ]

        if spec.event_complete or spec.label_ready or spec.loadable_only:
            allowed_primary_paths = set()
            selected_recordings = []
            candidate_paths = {f.path for f in candidates}
            for rec in recordings:
                if rec.primary.path not in candidate_paths:
                    continue
                if spec.event_complete and not rec.has_events:
                    continue
                if spec.label_ready and not rec.has_labels:
                    continue
                if spec.loadable_only and not rec.loadable:
                    continue
                allowed_primary_paths.add(rec.primary.path)
                selected_recordings.append(rec)
            companion_paths = {
                file.path
                for rec in selected_recordings
                for file in rec.companions.files
            }
            candidates = [
                f for f in candidates
                if f.path in allowed_primary_paths or f.path in companion_paths or f.is_essential
            ]
            if spec.label_ready and not allowed_primary_paths:
                warnings.append(
                    "label_ready=True requires confirmed label columns. "
                    "Manifest metadata alone only proves event-file candidates; "
                    "run check(local_path=...) or use event_complete=True for pre-download filtering."
                )

        # ── 9a. Exact-path include (set membership — no glob interpretation) ─
        if spec.exact_paths is not None:
            exact_set = set(spec.exact_paths)
            unmatched = exact_set - {f.path for f in candidates}
            if unmatched:
                for missing in sorted(unmatched):
                    suggestions = find_close_matches(missing, all_paths)
                    warnings.append(
                        f"Exact path {missing!r} not found in manifest"
                        + (f"; did you mean: {suggestions}" if suggestions else "")
                    )
            candidates = [f for f in candidates if f.path in exact_set]
            for file in candidates:
                add_reason(file, "exact path match")

        # ── 9b. Glob include ──────────────────────────────────────────────
        elif spec.include:
            candidates, included_set, _ = apply_include_exclude(
                candidates, spec.include, None
            )
            # Warn on patterns that matched nothing
            from qortex._internal.glob import glob_filter
            matched = glob_filter(all_paths, spec.include)
            for pattern, hits in matched.items():
                if not hits:
                    suggestions = find_close_matches(pattern, all_paths)
                    raise SelectionError(pattern, suggestions)
        else:
            # Default: exclude dotfiles
            candidates = [f for f in candidates if not is_dotfile(f.path)]

        for file in candidates:
            add_reason(file, "matched selection filters")

        # ── 10. Glob exclude ──────────────────────────────────────────────
        if spec.exclude:
            candidate_paths = [f.path for f in candidates]
            from qortex._internal.glob import glob_filter
            excl_matched = glob_filter(candidate_paths, spec.exclude)
            exclude_set: set[str] = {p for ms in excl_matched.values() for p in ms}
            # Never exclude essential files via glob
            essential_paths = {f.path for f in essential}
            exclude_set -= essential_paths
            candidates = [f for f in candidates if f.path not in exclude_set]

        selected_recordings = [
            rec for rec in recordings
            if rec.primary.path in {f.path for f in candidates}
        ]
        if spec.with_companions and not spec.metadata_only:
            before_paths = {f.path for f in candidates}
            expanded = graph.companion_closure(candidates)
            for file in expanded:
                if file.path not in before_paths:
                    rec = graph.recording_requiring_path(file.path)
                    add_reason(file, "required companion or inherited metadata", "manifest-graph", rec.id if rec else None)
            candidates = expanded

        if spec.max_size_gb is not None:
            max_bytes = int(spec.max_size_gb * 1e9)
            estimate = sum(f.size or 0 for f in candidates)
            if estimate > max_bytes:
                warnings.append(
                    f"Selection is ~{estimate / 1e9:.2f} GB, above max_size_gb={spec.max_size_gb:.2f}."
                )

        # ── 11. Merge + deduplicate ───────────────────────────────────────
        seen: set[str] = set()
        final: list[FileRecord] = []
        for f in essential + candidates:
            if f.path not in seen:
                seen.add(f.path)
                final.append(f)

        return final, essential, warnings, reasons, selected_recordings


def resolve_with_reasons(
    manifest: Manifest,
    spec: SelectionSpec,
) -> tuple[
    list[FileRecord],
    list[FileRecord],
    list[str],
    dict[str, list[SelectionReason]],
    list,
]:
    """Resolve a selection and include structured explanations."""
    return Selector().resolve_with_reasons(manifest, spec)
