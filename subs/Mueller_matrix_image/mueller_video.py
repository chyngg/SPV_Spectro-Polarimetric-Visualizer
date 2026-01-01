# mueller_video.py
import time
from fractions import Fraction

import numpy as np
import dearpygui.dearpygui as dpg

from subs import common_state
from subs.Mueller_matrix_image import mueller_state
from subs.Mueller_matrix_image.mueller_visualization import (
    visualize_rgb_mueller_grid,
    visualize_rgb_mueller_rgbgrid,
)

on_after_redraw = None


class MuellerVideoPlayer:
    """
    재생 알고리즘(끊김 방지, dpge 레이아웃 충돌 회피):
    - set_frame_callback이 덮어써지는 환경에서도 재생이 지속되도록 '배치 예약(batch scheduling)'을 사용
    - 예약 풀이 고갈되어 _step 호출이 끊겨도 'heartbeat'가 주기적으로 감지/복구
    - pause 누르기 전까지 계속 재생
    - 기본 동작: 마지막 프레임까지 재생 후 자동으로 stop (루프 X)
      (루프로 돌리고 싶으면 _advance_frame() 내부 주석 참고)
    - 6D 원본 비디오는 common_state.npy_video_data에 보관,
      현재 표시 프레임(5D)만 common_state.npy_data에 내려줌
    """

    def __init__(self) -> None:
        self.frames: np.ndarray | None = None  # (T,H,W,3,4,4)
        self.T: int = 0
        self.t: int = 0

        self.playing: bool = False
        self.fps_text: str = "10"

        # slider recursion guard
        self._ignore_slider_cb: bool = False

        # scheduler state
        self._last_step_time: float | None = None
        self._sched_running: bool = False

        # batch scheduling
        self._scheduled_frames: set[int] = set()
        self._batch_size: int = 400

        # heartbeat (스케줄 끊김 자동 복구)
        self._hb_running: bool = False
        self._hb_interval_frames: int = 15

        # detect video changes
        self._last_video_id: int | None = None

    # -------- helpers --------

    def parsed_fps(self) -> float:
        s = (self.fps_text or "10").strip()
        try:
            if "/" in s:
                return float(Fraction(s))
            return float(s)
        except Exception:
            return 10.0

    def _is_video_array(self, arr) -> bool:
        return (
            isinstance(arr, np.ndarray)
            and arr.ndim == 6
            and arr.shape[-3:] == (3, 4, 4)
            and arr.shape[0] > 0
        )

    def _get_video_source_array(self):
        arr6 = getattr(common_state, "npy_video_data", None)
        if self._is_video_array(arr6):
            return arr6
        arr = getattr(common_state, "npy_data", None)
        if self._is_video_array(arr):
            return arr
        return None

    def _set_slider_value_safely(self, v: int) -> None:
        if not dpg.does_item_exist("mueller_video_slider"):
            return
        self._ignore_slider_cb = True
        try:
            dpg.set_value("mueller_video_slider", int(v))
        finally:
            self._ignore_slider_cb = False

    def _ensure_video_attached(self) -> bool:
        if self.frames is not None and self.T > 0:
            return True

        arr6 = self._get_video_source_array()
        if arr6 is None:
            return False

        vid_id = id(arr6)
        if self._last_video_id != vid_id:
            self._last_video_id = vid_id
            self._scheduled_frames.clear()
            self._sched_running = False
            self.playing = False
            self._last_step_time = None
            self._hb_running = False

        self.frames = arr6
        self.T = int(arr6.shape[0])
        if self.T <= 0:
            self.frames = None
            self.T = 0
            self.t = 0
            return False

        self.t = max(0, min(self.t, self.T - 1))
        mueller_state.is_video = True

        # UI enable
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
            self._set_slider_value_safely(self.t)

        return True

    def has_video(self) -> bool:
        return self._ensure_video_attached()

    # -------- draw --------

    def redraw_current_frame(self) -> None:
        if not self.has_video():
            return

        data5d = self.frames[self.t]  # (H,W,3,4,4)
        common_state.npy_data = data5d

        if mueller_state.mueller_visualizing in ("Original", "m00", "Gamma"):
            visualize_rgb_mueller_grid(
                data5d,
                channel=mueller_state.mueller_selected_channel,
                correction=mueller_state.mueller_visualizing,
                vmin=common_state.vmin,
                vmax=common_state.vmax,
            )
        else:
            visualize_rgb_mueller_rgbgrid(
                data5d,
                correction=mueller_state.mueller_selected_correction,
                sign=mueller_state.mueller_visualizing,
            )

        if dpg.does_item_exist("mueller_video_frame_label"):
            dpg.set_value("mueller_video_frame_label", f"Frame: {self.t + 1}/{self.T}")

        self._set_slider_value_safely(self.t)

        if callable(on_after_redraw):
            try:
                on_after_redraw()
            except Exception as e:
                print("[mueller_video] on_after_redraw error:", repr(e))

    # -------- controls --------

    def start(self) -> None:
        if not self.has_video():
            return

        self.playing = True
        self._last_step_time = time.perf_counter()

        self._sched_running = False
        self._scheduled_frames.clear()

        self._schedule_next_step(force=True)
        self._start_heartbeat()

    def stop(self) -> None:
        self.playing = False
        self._last_step_time = None

        self._sched_running = False
        self._scheduled_frames.clear()

        self._hb_running = False

    def next_once(self) -> None:
        if not self.has_video():
            return
        self.stop()
        self.t = min(self.T - 1, self.t + 1)
        self.redraw_current_frame()

    def prev_once(self) -> None:
        if not self.has_video():
            return
        self.stop()
        self.t = max(0, self.t - 1)
        self.redraw_current_frame()

    def seek(self, idx: int) -> None:
        if not self.has_video():
            return
        self.stop()
        idx = int(idx)
        if 0 <= idx < self.T:
            self.t = idx
            self.redraw_current_frame()

    # -------- heartbeat --------

    def _start_heartbeat(self) -> None:
        if self._hb_running:
            return
        self._hb_running = True
        nxt = dpg.get_frame_count() + self._hb_interval_frames
        dpg.set_frame_callback(nxt, self._heartbeat)

    def _heartbeat(self, sender=None, app_data=None, user_data=None) -> None:
        if not self._hb_running or not self.playing:
            self._hb_running = False
            return

        if not self.has_video():
            self.stop()
            return

        if len(self._scheduled_frames) < (self._batch_size // 2):
            self._schedule_next_step(force=True)

        nxt = dpg.get_frame_count() + self._hb_interval_frames
        dpg.set_frame_callback(nxt, self._heartbeat)

    # -------- scheduler (batch scheduling) --------

    def _frame_interval(self) -> int:
        fps = max(1.0, float(self.parsed_fps()))
        try:
            render_fps = float(dpg.get_frame_rate())
            if render_fps <= 1.0:
                render_fps = 60.0
        except Exception:
            render_fps = 60.0

        interval = int(round(render_fps / fps))
        return max(4, interval)

    def _schedule_next_step(self, force: bool = False) -> None:
        if not self.playing:
            self._sched_running = False
            self._scheduled_frames.clear()
            return

        if self._sched_running and not force:
            return

        self._sched_running = True

        interval = self._frame_interval()
        base = dpg.get_frame_count()
        start = base + 2

        need = self._batch_size - len(self._scheduled_frames)
        if need <= 0:
            self._sched_running = False
            return

        k = 1
        while need > 0 and k <= self._batch_size * 20:
            f = start + k * interval
            if f not in self._scheduled_frames:
                self._scheduled_frames.add(f)
                dpg.set_frame_callback(f, self._step)
                need -= 1
            k += 1

        self._sched_running = False

    def _advance_frame(self, step: int) -> None:
        new_t = self.t + step

        if new_t >= self.T:
            self.t = self.T - 1
            self.redraw_current_frame()
            self.stop()
            return

        self.t = new_t
        self.redraw_current_frame()

    def _step(self, sender=None, app_data=None, user_data=None) -> None:
        try:
            frame_no = int(sender)
            self._scheduled_frames.discard(frame_no)
        except Exception:
            pass

        if not self.playing:
            self._sched_running = False
            self._scheduled_frames.clear()
            return

        if not self.has_video():
            self.stop()
            return

        now = time.perf_counter()
        fps = max(1.0, float(self.parsed_fps()))
        dt = 1.0 / fps

        last = self._last_step_time
        if last is None:
            last = now

        elapsed = now - last
        step = max(1, int(elapsed / dt))
        self._last_step_time = last + step * dt

        self._advance_frame(step)

        if self.playing:
            self._schedule_next_step(force=True)

    # -------- attach/detach --------

    def attach_frames(self, arr: np.ndarray) -> None:
        if not self._is_video_array(arr):
            raise ValueError(f"Expected (T,H,W,3,4,4), got {getattr(arr, 'shape', None)}")

        common_state.npy_video_data = arr  # 6D 원본 보관
        self.frames = arr
        self.T = int(arr.shape[0])
        self.t = 0
        self._last_video_id = id(arr)

        self.stop()

        mueller_state.is_video = True
        common_state.npy_data = arr[0]

        # UI enable
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
            self._set_slider_value_safely(0)

        if dpg.does_item_exist("mueller_video_frame_label"):
            dpg.set_value("mueller_video_frame_label", f"Frame: 1/{self.T}")

        self.redraw_current_frame()

    def detach_frames(self) -> None:
        self.stop()

        self.frames = None
        self.T = 0
        self.t = 0
        mueller_state.is_video = False
        self._last_video_id = None

        try:
            common_state.npy_video_data = None
        except Exception:
            pass

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


player = MuellerVideoPlayer()


# -------- callbacks --------

def cb_play(sender, app_data) -> None:
    player.start()

def cb_pause(sender, app_data) -> None:
    player.stop()

def cb_prev(sender, app_data) -> None:
    player.prev_once()

def cb_next(sender, app_data) -> None:
    player.next_once()

def cb_slider(sender, app_data) -> None:
    if player._ignore_slider_cb:
        return
    try:
        player.seek(int(app_data))
    except Exception:
        pass

def cb_fps(sender, app_data) -> None:
    player.fps_text = str(app_data or "").strip()

def on_mode_or_channel_changed() -> None:
    if mueller_state.is_video and player.has_video():
        player.redraw_current_frame()
