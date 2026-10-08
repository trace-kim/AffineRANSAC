"""All analysis viewers in one window, one tab each (pyqtgraph). Display only.

Tabs, for the merged contacts (registration.MergedErrors of one placement) and their RANSAC and
moving-window results:
- Stitching (optional): a StitchViewer built by the caller (needs the SEM images).
- RANSAC monitor (view_ransac.RansacMonitor), paused until Run is pressed.
- Registration, RANSAC: RegistrationView of the RANSAC residuals.
- Registration, moving window: RegistrationView of the moving-window residuals.
- Moving-window tuner (view_moving_window.MovingWindowTuner).
- Row pitch (view_pitch.PitchView): stitched without affine and after the RANSAC affine.
Every row profile also shows the stitched error without any affine (SEM − design) as a reference.

Optional extra result sets (D52, D55), e.g. the contacts after the in-image correction or after the
drift curve: each is drawn as extra lines (RegistrationView.add_rows) in both registration tabs (the
moving-window tab only if it has a moving-window result) and in the row pitch (after its RANSAC
affine), and gets one tab of its own: the registration view of its RANSAC residuals.

External reference measurements (add_external, e.g. another tool's result) go into every
registration view: lines in the row profile after the affine, markers on the error maps.

analysis_window(result) builds the whole window of an analysis.AnalysisResult, with the tabs the
notebooks add at the end (stitching, stripe drift curves, stitching residuals).
"""

import numpy as np
from pyqtgraph.Qt import QtWidgets

from affine_ransac.analysis import AnalysisResult, tile_edges_y
from affine_ransac.view_drift import StripeDriftView
from affine_ransac.view_moving_window import MovingWindowTuner
from affine_ransac.view_pitch import PitchView
from affine_ransac.view_points import PointsStitchView
from affine_ransac.view_ransac import RansacMonitor
from affine_ransac.view_registration import RegistrationView
from affine_ransac.view_stitch import StitchViewer
from affine_ransac.view_stitch_residuals import StitchResidualView


class AnalysisWindow(QtWidgets.QTabWidget):
    def __init__(
        self,
        sem_nm: np.ndarray,
        design_nm: np.ndarray,
        ransac,
        moving,
        window_um: float,
        step_um: float,
        threshold_nm: float = 0.5,
        seed: int = 0,
        delay_ms: int = 300,
        row_gap_nm: float = 10.0,
        tile_edges_y_nm=(),
        stitch_view: QtWidgets.QWidget | None = None,
        use_opengl: bool = True,
        extra: dict | None = None,
    ):
        """sem_nm, design_nm: (N, 2) merged contacts in mask nm (stitched, no affine); ransac: their
        RansacResult (ransac_affine with threshold_nm and seed); moving: their MovingWindowResult for
        window_um and step_um; delay_ms: RANSAC monitor pause per step; row_gap_nm: see group_rows;
        tile_edges_y_nm: y of the tile edges for the pitch plot; stitch_view: optional first tab;
        extra: optional {name: (sem_nm, design_nm, ransac, moving or None)}, other sets of contacts
        analysed with the same settings (e.g. corrected)."""
        super().__init__()
        self.setWindowTitle(f"AffineRANSAC analysis - {len(design_nm)} contacts")
        raw = sem_nm - design_nm  # stitched SEM − design, no affine (registration_error sign)
        reference = ransac.reference

        if stitch_view is not None:
            self.addTab(stitch_view, "Stitching")
        self.monitor = RansacMonitor(sem_nm, design_nm, threshold_nm, seed=seed, interval_ms=delay_ms,
                                     use_opengl=use_opengl)
        self.addTab(self.monitor, "RANSAC monitor")
        self.ransac_view = RegistrationView(design_nm, ransac.residuals, reference, row_gap_nm, use_opengl,
                                            correction="the RANSAC affine", raw_error_nm=raw, name="uncorrected")
        self.addTab(self.ransac_view, "Registration, RANSAC")
        self.moving_view = RegistrationView(
            design_nm, moving.residuals, reference, row_gap_nm, use_opengl,
            correction=f"the moving-window affine ({window_um:g} µm, step {step_um:g} µm)", raw_error_nm=raw,
            name="uncorrected")
        self.addTab(self.moving_view, "Registration, moving window")
        self.tuner = MovingWindowTuner(sem_nm, design_nm, ransac, window_um, step_um, row_gap_nm, use_opengl,
                                       raw_error_nm=raw)
        self.addTab(self.tuner, "Moving-window tuner")
        extra = extra or {}
        pitch_sets = {"stitched, no affine": sem_nm, "after RANSAC affine": design_nm + ransac.residuals}
        for name, (extra_sem, extra_design, extra_ransac, extra_moving) in extra.items():
            extra_raw = extra_sem - extra_design
            self.ransac_view.add_rows(extra_design, extra_ransac.residuals, name, extra_raw)
            if extra_moving is not None:
                self.moving_view.add_rows(extra_design, extra_moving.residuals, name, extra_raw)
            pitch_sets[f"{name}, after RANSAC affine"] = (extra_design, extra_design + extra_ransac.residuals)
        self.pitch_view = PitchView(design_nm, pitch_sets, reference, row_gap_nm, tile_edges_y_nm, use_opengl)
        self.addTab(self.pitch_view, "Row pitch")
        self.extra_views = {}
        for name, (extra_sem, extra_design, extra_ransac, _) in extra.items():
            view = RegistrationView(extra_design, extra_ransac.residuals, reference, row_gap_nm, use_opengl,
                                    correction=f"the RANSAC affine ({name})", raw_error_nm=extra_sem - extra_design,
                                    name=name)
            view.color_bar.setLevels(self.ransac_view.color_bar.levels())  # maps compare with the RANSAC tab
            self.extra_views[name] = view
            self.addTab(view, name[0].upper() + name[1:])
        self.resize(1700, 1050)

    def registration_views(self) -> list[RegistrationView]:
        """Every RegistrationView of the window: RANSAC, moving window, tuner, one per extra set."""
        return [self.ransac_view, self.moving_view, self.tuner.view, *self.extra_views.values()]

    def add_external(self, name: str, xy_nm: np.ndarray, error_nm: np.ndarray):
        """An external reference measurement in every registration view (RegistrationView.add_external):
        xy_nm (N, 2) site positions, mask nm; error_nm (N, 2) its dx, dy, nm."""
        for view in self.registration_views():
            view.add_external(name, xy_nm, error_nm)


def analysis_window(result: AnalysisResult, use_opengl: bool = True) -> AnalysisWindow:
    """The analysis window of a result, as the notebooks' last cell builds it: Stitching tab (the image
    stitch viewer after an image run, else the points view), the AnalysisWindow tabs with the corrected
    sets as extra lines and tabs, the subtracted drift bend in its tab, Stripe drift curves, Stitching
    residuals (translation and translation + rotation of every input)."""
    s = result.settings
    main, raw = result.sets["uncorrected"], result.stitchings["raw"]
    if result.images is not None:
        stitch_view = StitchViewer(result.images.tiles, result.tile_ids, result.images.design_polygons, raw.stitch,
                                   result.design_stitch, refined=s.stitch_refined, use_opengl=use_opengl,
                                   errors=result.images.merged, arrow_scale=s.arrow_scale,
                                   spread_flag_nm=s.spread_flag_nm)
    else:
        stitch_view = PointsStitchView(result.tile_ids, raw.points, result.design_points, result.centers, result.sizes,
                                       raw.stitch, result.design_stitch, use_opengl=use_opengl)
    extra = {name: (c.sem_nm, c.design_nm, c.ransac, c.moving) for name, c in result.sets.items() if name != "uncorrected"}
    window = AnalysisWindow(main.sem_nm, main.design_nm, main.ransac, main.moving, s.moving_window_um, s.moving_step_um,
                            s.ransac_threshold_nm, s.ransac_seed, s.ransac_delay_ms, s.row_gap_nm,
                            tile_edges_y(result.centers, result.sizes), stitch_view=stitch_view, use_opengl=use_opengl,
                            extra=extra)
    window.extra_views["drift corrected"].add_rows(main.design_nm, result.drift.correction(main.design_nm[:, 1]),
                                                   "drift correction applied")  # the bend that was subtracted
    reference = main.ransac.reference
    window.addTab(StripeDriftView(result.stripe_drift, reference, use_opengl=use_opengl), "Stripe drift curves")
    stitchings = {}
    for name, st in result.stitchings.items():
        stitchings[f"{name}, translation"] = (st.points, st.stitch, st.stitch.corrections)
        stitchings[f"{name}, translation + rotation"] = (st.points, st.stitch, st.rigid)
    window.addTab(StitchResidualView(stitchings, result.centers, reference, use_opengl=use_opengl), "Stitching residuals")
    return window
