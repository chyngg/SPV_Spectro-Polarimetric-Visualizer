import time
from fractions import Fraction
import threading

import numpy as np
import dearpygui.dearpygui as dpg

from subs import common_state
from subs.Mueller_matrix_image import mueller_state
from subs.Mueller_matrix_image.mueller_visualization import (
    visualize_rgb_mueller_grid,
    visualize_rgb_mueller_rgbgrid,
)


class MuellerVideoPlayer:

    def __init__(self) -> None:
        self.frames: np.ndarray | None = None  # (T,H,W,3,4,4)
        self.T: int = 0
        self.t: int = 0
        self.playing: bool = False
        self.fps_text: str = "10"

        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()


    def has_video(self) -> bool:
        return self.frames is not None and self.T > 0

    def attach_frames(self, arr: np.ndarray) -> None:
        if arr.ndim != 6 or arr.shape[-3:] != (3, 4, 4):
            raise ValueError(f"Expected (T,H,W,3,4,4), got {arr.shape}")

        self.frames = arr
        self.T = int(arr.shape[0])
        self.t = 0

        mueller_state.is_video = True

        common_state.npy_data = self.frames[0]

        if common_state.vmin >= common_state.vmax:
            common_state.vmin, common_state.vmax = -1.0, 1.0

        for tag in (
            "mueller_video_play",
            "mueller_video_pause",
            "mueller_video_prev",
            "mueller_video_next",
            "mueller_video_fps",
        ):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=True)

        if dpg.does_item_exist("mueller_video_slider"):
            dpg.configure_item(
                "mueller_video_slider",
                min_value=0,
                max_value=max(0, self.T - 1),
                enabled=True,
                show=True,
            )
            dpg.set_value("mueller_video_slider", 0)

        if dpg.does_item_exist("mueller_video_frame_label"):
            dpg.set_value("mueller_video_frame_label", f"Frame: 1/{self.T}")

        self.redraw_current_frame()

    def detach_frames(self) -> None:
        self.stop()
        self.frames = None
        self.T = 0
        self.t = 0
        mueller_state.is_video = False

        for tag in (
            "mueller_video_play",
            "mueller_video_pause",
            "mueller_video_prev",
            "mueller_video_next",
            "mueller_video_fps",
        ):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=False)

        if dpg.does_item_exist("mueller_video_frame_label"):
            dpg.set_value("mueller_video_frame_label", "")

        if dpg.does_item_exist("mueller_video_slider"):
            dpg.configure_item("mueller_video_slider", enabled=False, show=False)
            dpg.set_value("mueller_video_slider", 0)

    # ------------ playback helpers ------------

    def parsed_fps(self) -> float:
        s = (self.fps_text or "10").strip()
        try:
            if "/" in s:
                return float(Fraction(s))
            return float(s)
        except Exception:
            return 10.0

    def _advance_frame(self, delta: int) -> None:
        if not self.has_video():
            return
        self.t = (self.t + delta) % self.T
        self.redraw_current_frame()

    # ------------ public API used by callbacks ------------

    def redraw_current_frame(self) -> None:
        if not self.has_video():
            return

        common_state.npy_data = self.frames[self.t]

        if mueller_state.mueller_visualizing in ("Original", "m00", "Gamma"):
            visualize_rgb_mueller_grid(
                common_state.npy_data,
                channel=mueller_state.mueller_selected_channel,
                correction=mueller_state.mueller_visualizing,
                vmin=common_state.vmin,
                vmax=common_state.vmax,
            )
        else:
            visualize_rgb_mueller_rgbgrid(
                common_state.npy_data,
                correction=mueller_state.mueller_selected_correction,
                sign=mueller_state.mueller_visualizing,
            )

        if dpg.does_item_exist("mueller_video_frame_label"):
            dpg.set_value(
                "mueller_video_frame_label",
                f"Frame: {self.t + 1}/{self.T}",
            )

        if dpg.does_item_exist("mueller_video_slider"):
            dpg.set_value("mueller_video_slider", self.t)

    def play_once(self) -> None:
        self._advance_frame(+1)

    def prev_once(self) -> None:
        self._advance_frame(-1)

    # ------------ background loop ------------

    def _loop(self) -> None:
        while not self._stop_evt.is_set() and self.playing and self.has_video():
            fps = max(1e-3, self.parsed_fps())
            start = time.time()
            self._advance_frame(+1)
            elapsed = time.time() - start
            wait = max(0.0, 1.0 / fps - elapsed)
            time.sleep(wait)

    def start(self) -> None:
        if not self.has_video():
            return
        self.playing = True
        self._stop_evt.clear()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self.playing = False
        self._stop_evt.set()


player = MuellerVideoPlayer()



def cb_play(sender, app_data) -> None:
    player.start()


def cb_pause(sender, app_data) -> None: # noqa: ARG001
    """Pause 버튼."""
    player.stop()


def cb_prev(sender, app_data) -> None:
    player.prev_once()


def cb_next(sender, app_data) -> None:
    player.play_once()


def cb_fps(sender, app_data) -> None:
    player.fps_text = str(app_data or "").strip()


def on_mode_or_channel_changed() -> None:
    if not mueller_state.is_video or not player.has_video():
        return
    player.redraw_current_frame()

def cb_slider(sender, app_data) -> None:
    if not player.has_video():
        return
    try:
        idx = int(app_data)
    except Exception:
        return

    if idx < 0 or idx >= player.T:
        return

    player.t = idx
    player.redraw_current_frame()

