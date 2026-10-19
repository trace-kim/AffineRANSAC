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
notebooks add at the end (stitching, stripe drift curves, stitching residuals), and Contact size (area,
xCD, yCD maps of an image run, measured on request; N/A for contour CSVs). Its Stitching tab
(StitchingTab) has a Stitching method box: choosing another method calls on_restitch(method), e.g. the
app's analysis.restitch in the background, which then shows the new result in a new window.
"""

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from affine_ransac.analysis import STITCH_METHODS, AnalysisResult, tile_edges_y
from affine_ransac.view_drift import StripeDriftView
from affine_ransac.view_contact_size import ContactSizeTab
from affine_ransac.view_moving_window import MovingWindowTuner
from affine_ransac.view_pitch import PitchView
from affine_ransac.view_points import PointsStitchView
from affine_ransac.view_ransac import RansacMonitor
from affine_ransac.view_registration import EXTRA_COLORS, RegistrationView
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
        for k, (name, (extra_sem, extra_design, extra_ransac, extra_moving)) in enumerate(extra.items()):
            extra_raw = extra_sem - extra_design
            colors = EXTRA_COLORS[k % len(EXTRA_COLORS)]  # a set has the same colours in both tabs
            self.ransac_view.add_rows(extra_design, extra_ransac.residuals, name, extra_raw, colors)
            if extra_moving is not None:
                self.moving_view.add_rows(extra_design, extra_moving.residuals, name, extra_raw, colors)
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


class StitchingTab(QtWidgets.QWidget):
    def __init__(self, view: QtWidgets.QWidget, method: str, on_restitch=None, unavailable: str = ""):
        """The stitch view (StitchViewer or PointsStitchView) under a Stitching method box set to `method`.
        Choosing another method calls on_restitch(method); without on_restitch, or with an `unavailable`
        reason, the box is disabled (its tooltip says why)."""
        super().__init__()
        self.view, self.method, self.on_restitch = view, method, on_restitch
        self.methods = QtWidgets.QComboBox()
        self.methods.addItems(STITCH_METHODS)
        self.methods.setCurrentText(method)
        self.methods.setToolTip("all overlaps: every overlap paired at the nominal placement.\n"
                                "measurement order first: the images taken one after the other are stitched "
                                "first; every overlap is then paired at the offset that predicts, so drift "
                                "between images taken far apart cannot pair contacts one row off.\n"
                                "Changing it computes the stitching and every result after it again.")
        reason = unavailable or ("" if on_restitch is not None else "Change stitch_method in the settings")
        if reason:
            self.methods.setEnabled(False)
            self.methods.setToolTip(reason)
        self.methods.currentTextChanged.connect(self._chosen)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Stitching method:"))
        bar.addWidget(self.methods)
        bar.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(view, stretch=1)

    def _chosen(self, method: str):
        if method != self.method:
            self.methods.setEnabled(False)  # until the new result is shown (a new window) or reset_method
            self.on_restitch(method)

    def reset_method(self):
        """Back to the method this window shows (e.g. the restitching failed)."""
        self.methods.blockSignals(True)
        self.methods.setCurrentText(self.method)
        self.methods.blockSignals(False)
        self.methods.setEnabled(True)


def analysis_window(result: AnalysisResult, use_opengl: bool = True, on_restitch=None) -> AnalysisWindow:
    """The analysis window of a result, as the notebooks' last cell builds it: Stitching tab (the image
    stitch viewer after an image run, else the points view), the AnalysisWindow tabs with the corrected
    sets as extra lines and tabs, the subtracted drift bend in its tab, Stripe drift curves, Stitching
    residuals (translation and translation + rotation of every input). on_restitch(method): see StitchingTab;
    window.stitching_tab is that tab."""
    s = result.settings
    main, raw = result.sets["uncorrected"], result.stitchings["raw"]
    if result.images is not None:
        stitch_view = StitchViewer(result.images.tiles, result.tile_ids, result.images.design_polygons, raw.stitch,
                                   result.design_stitch, use_opengl=use_opengl,
                                   errors=result.images.merged, arrow_scale=s.arrow_scale,
                                   spread_flag_nm=s.spread_flag_nm)
    else:
        stitch_view = PointsStitchView(result.tile_ids, raw.points, result.design_points, result.centers, result.sizes,
                                       raw.stitch, result.design_stitch, use_opengl=use_opengl)
    extra = {name: (c.sem_nm, c.design_nm, c.ransac, c.moving) for name, c in result.sets.items() if name != "uncorrected"}
    unavailable = ("Restitching needs the SEM images in memory: analyse the image folder again"
                   if result.kind == "images" and result.images is None else "")
    stitching_tab = StitchingTab(stitch_view, s.stitch_method, on_restitch, unavailable)
    window = AnalysisWindow(main.sem_nm, main.design_nm, main.ransac, main.moving, s.moving_window_um, s.moving_step_um,
                            s.ransac_threshold_nm, s.ransac_seed, s.ransac_delay_ms, s.row_gap_nm,
                            tile_edges_y(result.centers, result.sizes), stitch_view=stitching_tab, use_opengl=use_opengl,
                            extra=extra)
    window.stitching_tab = stitching_tab
    window.extra_views["drift corrected"].add_rows(main.design_nm, result.drift.correction(main.design_nm[:, 1]),
                                                   "drift correction applied")  # the bend that was subtracted
    reference = main.ransac.reference
    window.addTab(StripeDriftView(result.stripe_drift, reference, use_opengl=use_opengl), "Stripe drift curves")
    stitchings = {}
    for name, st in result.stitchings.items():
        stitchings[f"{name}, translation"] = (st.points, st.stitch, st.stitch.corrections)
        stitchings[f"{name}, translation + rotation"] = (st.points, st.stitch, st.rigid)
    window.addTab(StitchResidualView(stitchings, result.centers, reference, use_opengl=use_opengl), "Stitching residuals")
    if result.kind == "images":
        sizes = ContactSizeTab(result, use_opengl=use_opengl)
    else:
        sizes = QtWidgets.QLabel("N/A: contour CSVs hold contact centres only. Area, xCD and yCD are measured on "
                                 "the SEM contours and .oas outlines of an image folder.")
        sizes.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    window.addTab(sizes, "Contact size")
    return window
