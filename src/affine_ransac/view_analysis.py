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
"""

import numpy as np
from pyqtgraph.Qt import QtWidgets

from affine_ransac.view_moving_window import MovingWindowTuner
from affine_ransac.view_pitch import PitchView
from affine_ransac.view_ransac import RansacMonitor
from affine_ransac.view_registration import RegistrationView


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
