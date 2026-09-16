"""Notebook workflows for inspecting and comparing saved hardware snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from qgan_v2.analysis.backend_transpilation import BackendTranspilation
    from qgan_v2.analysis.backend_placement import BackendPlacementComparison

from qgan_v2.analysis.backend_data import (
    QUBIT_ERROR_METRICS,
    QUBIT_TIME_METRICS,
    backend_layout_description,
    fmt,
    gate_calibration_summary,
    load_backend_snapshot,
    plot_chip_layout,
    plot_gate_error_comparison,
    plot_gate_time_comparison,
    plot_qubit_error_comparison,
    plot_qubit_time_comparison,
    qubit_calibration_summary,
    resolve_backend_path,
)
from qgan_v2.analysis.results import save_figure


@dataclass
class BackendComparison:
    """Loaded snapshots, calibration summaries, and any exported comparison images."""

    snapshots: dict[str, dict[str, Any]]
    qubit_summaries: dict[str, list[dict[str, Any]]]
    gate_summaries: dict[str, list[dict[str, Any]]]
    output_files: dict[str, Path]


def _display_calibration_table(title: str, rows: list[dict[str, Any]]) -> None:
    from IPython.display import HTML, display

    columns = list(rows[0]) if rows else []
    head = ''.join(f'<th>{html.escape(c)}</th>' for c in columns)
    body = ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(fmt(r.get(c)))}</td>' for c in columns) + '</tr>'
        for r in rows
    )
    display(HTML(
        f'<p><b>{html.escape(title)}</b></p><div style="overflow:auto">'
        f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    ))


class BackendAnalysis:
    """Inspect trusted backend pickle files without contacting IBM Runtime.

    Relative snapshot paths are resolved against ``repo_root``. Loaded snapshots
    are reused within this analysis instance; use ``refresh=True`` to reread a
    file whose content has changed.
    """

    def __init__(
        self,
        repo_root: str | Path,
        *,
        figure_dir: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.figure_dir = (
            resolve_backend_path(figure_dir, self.repo_root)
            if figure_dir is not None
            else self.repo_root / 'qgan/figures/rh_backend_data'
        )
        self._snapshots: dict[Path, dict[str, Any]] = {}
        self._placement_transpilations: dict[tuple, BackendTranspilation] = {}
        self._fake_placement_snapshot: dict[str, Any] | None = None

    def load_backend(
        self, file: str | Path, *, refresh: bool = False,
        save_figures: bool = False, filename: str | None = None,
    ) -> dict[str, Any]:
        """Load/reuse a snapshot and optionally save its calibrated chip layout.

        ``save_figures=True`` exports one 300-dpi PNG in ``figure_dir``, without
        displaying it. ``filename`` is an optional file stem; the default uses
        the snapshot stem and property-update date. Export also works when the
        snapshot is cached. Use ``show_layout`` to display the loaded snapshot.
        """
        if filename is not None and (not filename or Path(filename).name != filename
                                     or filename in ('.', '..')):
            raise ValueError('filename must be a nonempty file stem without directories.')
        path = resolve_backend_path(file, self.repo_root)
        if refresh or path not in self._snapshots:
            self._snapshots[path] = load_backend_snapshot(path, self.repo_root)
        snapshot = self._snapshots[path]
        if save_figures:
            import matplotlib.pyplot as plt

            date = getattr(snapshot['data']['properties'], 'last_update_date', None)
            stem = filename if filename is not None else (
                f'backend_layout_{path.stem}_{date.date().isoformat() if date is not None else "undated"}')
            with plt.rc_context({'font.size': 10, 'figure.dpi': 120}):
                fig = plot_chip_layout(snapshot, self.repo_root)
                try:
                    saved = save_figure(fig, self.figure_dir, stem, formats=('png',), dpi=300)
                    print('Backend layout PNG saved:', saved[0])
                finally:
                    plt.close(fig)
        return snapshot

    def show_layout(self, snapshot: dict[str, Any]) -> None:
        """Display the selected chip description and calibrated layout."""
        import matplotlib.pyplot as plt
        from IPython.display import Markdown, display

        with plt.rc_context({'font.size': 10, 'figure.dpi': 120}):
            fig = plot_chip_layout(snapshot, self.repo_root)
            try:
                display(Markdown(backend_layout_description(snapshot)))
                plt.show()
            finally:
                plt.close(fig)

    def show_calibration_summary(
        self, snapshot: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Display and return the qubit and gate calibration summaries."""
        qubits = qubit_calibration_summary(snapshot)
        gates = gate_calibration_summary(snapshot)
        _display_calibration_table('Qubit calibration summary', qubits)
        _display_calibration_table('Gate calibration summary', gates)
        return qubits, gates

    def show_backend(self, file: str | Path) -> dict[str, Any]:
        """Load a backend and display its layout and both calibration summaries."""
        snapshot = self.load_backend(file)
        self.show_layout(snapshot)
        self.show_calibration_summary(snapshot)
        return snapshot

    def compare_errors(
        self,
        files: Mapping[str, str | Path],
        *,
        save_figures: bool = False,
        filename: str | None = None,
    ) -> BackendComparison:
        """Display error figures and optionally export 300-dpi PNGs.

        ``filename`` sets a shared stem, exporting ``{filename}_qubit.png``
        and ``{filename}_gate.png``. None preserves the default export names.
        """
        return self._compare_calibrations(
            files, comparison_type='errors', save_figures=save_figures, filename=filename)

    def compare_times(
        self,
        files: Mapping[str, str | Path],
        *,
        save_figures: bool = False,
        filename: str | None = None,
    ) -> BackendComparison:
        """Compare qubit relaxation/coherence/readout times and gate durations.

        Qubit plots display all times in microseconds; the returned summaries
        preserve their labelled units (readout duration in nanoseconds). Gate
        durations are in nanoseconds. T1/T2 are not operation durations.
        ``filename`` sets a shared stem for ``{filename}_qubit.png`` and
        ``{filename}_gate.png``; None preserves the default export names.
        """
        return self._compare_calibrations(
            files, comparison_type='times', save_figures=save_figures, filename=filename)

    def reconstruct_transpilation(
        self, snapshot: dict[str, Any], config_file: str | Path,
        *, save_directory: str | Path | None = None, show_summary: bool = True,
        preset: str | None = None,
        default_layout: bool = False,
    ) -> BackendTranspilation:
        """Display offline template placement/depth/CZ counts without IBM access.

        The caller chooses the matching run config explicitly: a shared noisy
        snapshot alone cannot identify which run or circuit to reconstruct.
        ``default_layout`` overrides only this offline compilation.
        """
        from IPython.display import Markdown, display
        from qgan_v2.analysis.backend_transpilation import (
            reconstruct_backend_transpilation, save_backend_transpilation, transpilation_summary,
        )

        result = reconstruct_backend_transpilation(
            snapshot, resolve_backend_path(config_file, self.repo_root), preset=preset,
            default_layout=default_layout)
        if show_summary:
            display(Markdown(
                '**Offline reconstruction — historical job circuits are not verified.** '
                'Depth and CZ counts describe unbound training templates, before gradient '
                'and Runtime measurement/mitigation expansion. Initial and final placements '
                'list physical indices in logical-qubit order.'))
            _display_calibration_table('Reconstructed training circuit summary', transpilation_summary(result))
            print('Transpilation settings:', result.provenance['settings'])
            print('Software versions:', result.provenance['software_versions'])
        if save_directory is not None:
            directory = resolve_backend_path(save_directory, self.repo_root)
            save_backend_transpilation(result, directory)
            if show_summary:
                print('Reconstructed JSON and QPY files saved in:', directory)
        return result

    def show_transpilation_layout(
        self, snapshot: dict[str, Any], result: BackendTranspilation,
        *, template_index: int = 0, save_figures: bool = False, title: str | None = None,
    ) -> Path | None:
        """Show one reconstructed template on the full chip using the packed plotter.

        Node labels show circuit qubits at their initial physical placement.
        Additional wires touched during routing are highlighted separately.
        The final logical mapping remains available in the summary table.
        """
        import matplotlib.pyplot as plt
        from IPython.display import Markdown, display
        from qgan_v2.visualization.hardware_layout import plot_hardware_layout
        from qgan_v2.analysis.backend_placement import placement_layout_job

        if not 0 <= template_index < len(result.records):
            raise ValueError(f'template_index must be between 0 and {len(result.records) - 1}.')
        source_file = result.provenance['backend_file']
        if ((source_file is None) != (snapshot['path'] is None)
                or source_file is not None and Path(source_file).resolve() != snapshot['path'].resolve()):
            raise ValueError('The reconstruction and layout must use the same backend snapshot.')
        record = result.records[template_index]
        job = placement_layout_job(result.transpiled_circuits[template_index], record)
        display(Markdown(
            '**Reconstructed physical-qubit placement.** Coloured nodes show the '
            'initial placement; labels give the circuit qubit (q0, q1, …). '
            'Grey nodes are idle; black links are used by the circuit. '
            'Any additional active qubits are orange and labelled aux. The summary table contains '
            'the final mapping after routing. Historical job placement is unverified.'))
        output = None
        with plt.rc_context({'font.size': 10, 'figure.dpi': 120}):
            fig = plot_hardware_layout(job, snapshot['data'],
                title if title is not None else 'Offline reconstructed physical-qubit placement')
            try:
                if save_figures:
                    output = save_figure(
                        fig, self.figure_dir, f'reconstructed_qubit_placement_{template_index}',
                        formats=('png',), dpi=300,
                    )[0]
                plt.show()
            finally:
                plt.close(fig)
        return output

    def compare_transpilations(
        self, results: Mapping[str, BackendTranspilation],
    ) -> list[dict[str, Any]]:
        """Display one compact comparison of already reconstructed templates."""
        if not results:
            raise ValueError('Choose at least one reconstruction to compare.')
        rows = [
            {'Reference': label, 'Template': record['template'],
             'Depth': record['depth'], '2-qubit depth': record['two_qubit_depth'],
             'CZ count': record['cz_count'], 'Active qubits': len(record['active_physical_qubits'])}
            for label, result in results.items() for record in result.records
        ]
        _display_calibration_table('Offline reconstructed template comparison', rows)
        return rows

    def show_transpilation_layouts(
        self, *, n_qubits=(4, 8, 16),
        circuit=('generator-discriminator', 'real-discriminator'), backend='noisy',
        implementation='qml-torch', preset='base',
        save_figures: bool = False, filename: str | None = None, title: str | None = None,
        backend_files: Mapping[str, str | Path] | None = None,
        config_files: Mapping[str, Mapping[int, str | Path]] | None = None,
        real_circuit_index: int = 0,
        default_layout: bool = False,
    ) -> BackendPlacementComparison:
        """Display one combined figure for every requested placement combination.

        Scalars and tuples/lists are accepted. Multiple sizes become columns;
        a single size makes backends the columns. Rows group backend/circuit
        pairs or circuits respectively. Templates are compiled once per
        backend/size/implementation/preset and reused across calls within this instance.

        Defaults use saved noisy q4/q8/q16 configs and real q4 configs. Larger
        real sizes are explicit offline variants of the q4 settings and are
        labelled accordingly. No IBM service or training execution is created.
        No tables are displayed. Optional exports use one 300-dpi PNG.
        With one placement, only ``title`` is shown; multiple placements also
        have automatic backend/circuit/size panel headings.
        ``fake-real`` loads bundled FakeSherbrooke. ``implementation`` selects
        QML, packed separate, or packed joined circuit construction; it may be
        a scalar or sequence and participates in the Cartesian product.
        ``generator`` and ``discriminator`` are aliases for the two existing
        circuit selectors. Joined panels name the independently compiled
        generator (fake→D) and discriminator (real→D plus fake→D) jobs.
        Each real/fake copy has a distinct, consistent legend color.
        ``preset`` selects base (direct), ang (angle), or amp (amplitude) and
        also accepts a scalar or sequence. Unsupported combinations (including amp
        with packed joined), source/compilation failures, and unavailable templates
        are skipped with reported reasons in ``skipped_combinations``. Remaining
        panels are packed into the figure; if none succeed, no figure is created.
        Switching to joined angle rounds an odd batch size up to an even one.
        Preset variants keep the source's ansatz, randomness, and compiler settings;
        preset, dataset, encoding, and batch changes are recorded in provenance.
        ``default_layout`` selects Qiskit's default layout pipeline for these
        offline variants without changing the saved run configurations.
        """
        import matplotlib.pyplot as plt
        from IPython.display import Markdown, display
        from qgan_v2.analysis.backend_placement import (
            BACKEND_PLACEMENT_FILES, BACKEND_PLACEMENT_CONFIGS,
            PACKED_PLACEMENT_CONFIGS, fake_backend_snapshot,
            BackendPlacement, BackendPlacementComparison,
            placement_combinations, plot_backend_placements,
        )
        from qgan_v2.analysis.backend_transpilation import reconstruct_backend_transpilation

        combinations, columns = placement_combinations(n_qubits, circuit, backend, implementation, preset)
        files = BACKEND_PLACEMENT_FILES if backend_files is None else backend_files
        if (not isinstance(real_circuit_index, int) or isinstance(real_circuit_index, bool)
                or real_circuit_index < 0):
            raise ValueError('real_circuit_index must be a nonnegative integer.')
        if filename is not None and (not filename or Path(filename).name != filename
                                     or filename in ('.', '..')):
            raise ValueError('filename must be a nonempty file stem without directories.')
        # A failed configuration affects its panels, not the other combinations.
        sources = {}
        failures = {}
        placements = []
        skipped = []
        for b, c, n, impl, p in combinations:
            source_key = (b, n, impl, p)
            if source_key not in sources and source_key not in failures:
                try:
                    if impl == 'runtime-packed-join' and p == 'amp':
                        raise ValueError('Amplitude encoding is unsupported by runtime-packed-join; '
                                         'use qml-torch or runtime-packed-sep.')
                    configs = (config_files if config_files is not None else
                               {**BACKEND_PLACEMENT_CONFIGS, **PACKED_PLACEMENT_CONFIGS.get(impl, {})})
                    if (b != 'fake-real' and b not in files) or b not in configs:
                        raise ValueError(f'Provide a backend file and run configs for {b!r}.')
                    config_file = configs[b].get(n, configs[b].get(4))
                    if config_file is None:
                        raise ValueError(f'Provide a run config for {b!r} with {n} qubits.')
                    path = resolve_backend_path(config_file, self.repo_root)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    if b == 'fake-real':
                        if self._fake_placement_snapshot is None:
                            self._fake_placement_snapshot = fake_backend_snapshot()
                        snapshot = self._fake_placement_snapshot
                    else:
                        snapshot = self.load_backend(files[b])
                    stat = path.stat()
                    key = (id(snapshot), path, stat.st_mtime_ns, stat.st_size, n, impl, p,
                           default_layout)
                    if key not in self._placement_transpilations:
                        print(f'Reconstructing {b}, {n} qubits, {impl}, {p}...')
                        self._placement_transpilations[key] = reconstruct_backend_transpilation(
                            snapshot, path, n_qubits=n, implementation=impl, preset=p,
                            default_layout=default_layout)
                    sources[source_key] = (snapshot, self._placement_transpilations[key])
                except Exception as error:
                    failures[source_key] = f'{type(error).__name__}: {error}'
            reason = failures.get(source_key)
            if reason is None:
                snapshot, result = sources[source_key]
                index = 0 if c == 'generator-discriminator' else real_circuit_index + 1
                if index >= len(result.records):
                    reason = f'Real circuit index {real_circuit_index} is unavailable.'
                else:
                    placements.append(BackendPlacement(b, n, c, index, snapshot, result, impl, p))
            if reason is not None:
                skipped.append({'backend': b, 'circuit': c, 'n_qubits': n,
                                'implementation': impl, 'preset': p, 'reason': reason})
                print(f'Skipped {b}, {n} qubits, {impl}, {p}, {c}: {reason}')
        if not placements:
            raise ValueError(f'No placement could be reconstructed; all {len(skipped)} combinations '
                             'were skipped. See the reported reasons above.')
        comparison = BackendPlacementComparison(placements, skipped_combinations=skipped)
        display(Markdown(
            f'**{len(placements)} placements joined in one figure; {len(skipped)} skipped.** Labels show circuit '
            'qubits (q0, q1, …) at initial placement; black links are used by the transpiled '
            'circuit. These are offline reconstructions, with full initial/final mappings '
            'available in the returned circuit records.'))
        with plt.rc_context({'font.size': 9, 'figure.dpi': 120}):
            fig = plot_backend_placements(placements, columns, title=title)
            try:
                if save_figures:
                    if filename is None:
                        backends = '-'.join(dict.fromkeys(p.backend for p in placements))
                        sizes = '-'.join(str(n) for n in dict.fromkeys(p.n_qubits for p in placements))
                        circuits = '-'.join(dict.fromkeys(p.circuit for p in placements))
                        implementations = '-'.join(dict.fromkeys(p.implementation for p in placements))
                        presets = '-'.join(dict.fromkeys(p.preset for p in placements))
                        filename = f'qubit_placements_{backends}_q{sizes}_{circuits}_{implementations}_{presets}'
                    saved = save_figure(fig, self.figure_dir, filename, formats=('png',), dpi=300)
                    comparison.output_files = {p.suffix.lstrip('.'): p for p in saved}
                plt.show()
            finally:
                plt.close(fig)
        return comparison

    def _compare_calibrations(
        self,
        files: Mapping[str, str | Path],
        *,
        comparison_type: str,
        save_figures: bool,
        filename: str | None = None,
    ) -> BackendComparison:
        import matplotlib.pyplot as plt

        if not files:
            raise ValueError('Choose at least one backend file to compare.')
        if filename is not None and (not filename or Path(filename).name != filename
                                     or filename in ('.', '..')):
            raise ValueError('filename must be a nonempty file stem without directories.')
        if comparison_type == 'times':
            qubit_metrics = QUBIT_TIME_METRICS
            gate_metric = 'Duration (ns)'
            plots = (('qubit', plot_qubit_time_comparison), ('gate', plot_gate_time_comparison))
            suffix = 'time_calibration_comparison'
        else:
            qubit_metrics = QUBIT_ERROR_METRICS
            gate_metric = 'Error (%)'
            plots = (('qubit', plot_qubit_error_comparison), ('gate', plot_gate_error_comparison))
            suffix = 'calibration_comparison'
        snapshots = {label: self.load_backend(file) for label, file in files.items()}
        qubit_summaries = {
            label: [r for r in qubit_calibration_summary(s) if r['Metric'] in qubit_metrics]
            for label, s in snapshots.items()
        }
        gate_summaries = {
            label: [r for r in gate_calibration_summary(s) if r['Metric'] == gate_metric]
            for label, s in snapshots.items()
        }
        output_files = {}
        with plt.rc_context({'font.size': 10, 'figure.dpi': 120}):
            for kind, plot in plots:
                fig = plot(snapshots)
                try:
                    if save_figures:
                        saved = save_figure(
                            fig, self.figure_dir,
                            f'{filename}_{kind}' if filename is not None else f'{kind}_{suffix}',
                            formats=('png',), dpi=300,
                        )
                        output_files[kind] = saved[0]
                    plt.show()
                finally:
                    plt.close(fig)
        print('All comparison snapshots passed completeness and unit-conversion checks.')
        if output_files:
            print('Comparison images saved in:', self.figure_dir)
        return BackendComparison(snapshots, qubit_summaries, gate_summaries, output_files)
