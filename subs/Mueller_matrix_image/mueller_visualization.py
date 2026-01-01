from subs.Mueller_matrix_image import mueller_state
from subs import common_state
import dearpygui.dearpygui as dpg
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def _stitch_mueller_4x4_scalar(npy_data: np.ndarray, channel: int) -> np.ndarray:
	tiles = npy_data[:, :, channel, :, :]  # (H, W, 4, 4)

	rows = []
	for i in range(4):
		row_tiles = [tiles[:, :, i, j] for j in range(4)]           # 4개의 (H,W)
		row_strip = np.concatenate(row_tiles, axis=1)                # (H, W*4)
		rows.append(row_strip)
	big_scalar = np.concatenate(rows, axis=0)                        # (H*4, W*4)
	return big_scalar

def _stitch_mueller_4x4_rgb(rgb_4x4: np.ndarray) -> np.ndarray:
	rows = []
	for i in range(4):
		row_tiles = [rgb_4x4[:, :, :, i, j] for j in range(4)]        # 4개의 (H,W,3)
		row_strip = np.concatenate(row_tiles, axis=1)                  # (H, W*4, 3)
		rows.append(row_strip)
	big_rgb = np.concatenate(rows, axis=0)                             # (H*4, W*4, 3)
	return big_rgb

def visualize_rgb_mueller_grid(data5d: np.ndarray, channel: str="R", correction : str="Original", vmin: float = -1.0, vmax: float = 1.0):
	channel_num = ("B", "G", "R").index(channel)
	mueller_state.mueller_visualize_rgb = False
	tiles = data5d[:, :, channel_num, :, :]  # (H, W, 4, 4)
	tiles = _apply_mueller_correction(tiles, mode=correction)

	def _stitch_mueller_4x4_scalar(npy_data_4x4: np.ndarray) -> np.ndarray:
		# npy_data_4x4: (H, W, 4, 4)
		rows = []
		for i in range(4):
			row_tiles = [npy_data_4x4[:, :, i, j] for j in range(4)]
			row_strip = np.concatenate(row_tiles, axis=1)  # (H, W*4)
			rows.append(row_strip)
		big_scalar_local = np.concatenate(rows, axis=0)   # (H*4, W*4)
		return big_scalar_local

	big_scalar = _stitch_mueller_4x4_scalar(tiles)

	title_mode = {
		"Original": "Original",
		"Gamma": f"Gamma (γ={mueller_state.gamma:.3f})",
		"m00": "m00-Normalized"
	}.get(correction, correction)

	return generate_texture(
		image_data=big_scalar,
		title=f"Mueller-Matrix 4x4 ({channel} Channel) - {title_mode}",
		colormap="RdBu",
		vmin=vmin,
		vmax=vmax,
		is_original=False,
		normalize=False,
	)

def _apply_mueller_correction(mat4x4: np.ndarray, mode: str = "Original", eps: float = 1e-6) -> np.ndarray:
	"""
	mat4x4: (H, W, 4, 4) Mueller matrix per-pixel
	mode: "original" | "gamma" | "m00"
	- gamma: signed gamma -> sign(x) * |x|**gamma
	- m00:   elementwise divide by |m00| (per pixel)
	"""
	if mode == "Gamma":
		mueller_state.visualizing_gamma = True
		return (np.abs(mat4x4) ** (1 / mueller_state.gamma)) * np.sign(mat4x4)

	elif mode == "m00":
		m00 = mat4x4[:, :, 0, 0]
		denom = np.maximum(np.abs(m00), eps)
		mueller_state.visualizing_gamma = False
		return mat4x4 / denom[:, :, None, None]

	else:
		mueller_state.visualizing_gamma = False
		return mat4x4

def visualize_rgb_mueller_rgbgrid(data5d: np.ndarray, correction: str,  sign: str): # Positive / Negative
	if data5d is None:
		return
	mueller_state.mueller_visualize_rgb = True
	rgb_4x4 = _apply_mueller_correction(data5d, mode=correction)

	if sign == "Negative":
		rgb_4x4 = np.clip(rgb_4x4, -1, 0) * (-1)
		title = "Negative"
	else: # Positive
		rgb_4x4 = np.clip(rgb_4x4, 0, 1)
		title = "Positive"

	big_rgb = _stitch_mueller_4x4_rgb(rgb_4x4)  # (H*4, W*4, 3)
	title_mode = {
		"Original": "Original",
		"Gamma": f"Gamma (γ={mueller_state.gamma:.3f})",
		"m00": "m00-Normalized"
	}.get(correction, correction)

	return generate_texture(
		image_data=big_rgb,
		title=f"{title} - {title_mode}",
		is_original=True
	)

def generate_texture(image_data, title, colormap=None, vmin=None, vmax=None, is_original=False, normalize=None):
	if normalize is None:
		normalize = (vmin is None and vmax is None and not is_original)
	fig, ax = plt.subplots(figsize=(7.6, 5.4))
	try:
		h, w = common_state.npy_data.shape[:2]
		if is_original:
			display_image = cv2.resize(image_data, (760, 540), interpolation=cv2.INTER_LINEAR)
			img = ax.imshow(display_image)

		elif image_data.ndim == 2:
			display_image = cv2.resize(image_data, (760, 540), interpolation=cv2.INTER_LINEAR)
			if normalize:
				rng = np.ptp(display_image)
				if rng > 0: display_image = (display_image-np.min(display_image)) / (rng + 1e-8)
			img = ax.imshow(display_image, cmap=colormap, interpolation="nearest", vmin=vmin, vmax=vmax)
		else:
			display_image = cv2.resize(np.mean(image_data, axis=2), (760, 540), interpolation=cv2.INTER_LINEAR)
			if normalize:
				rng = np.ptp(display_image)
				if rng > 0: display_image = (display_image - np.min(display_image)) / (rng + 1e-8)
			img = ax.imshow(display_image, cmap=colormap, interpolation="nearest", vmin=vmin, vmax=vmax)

		ax.axis("on")
		ax.set_title(title)

		if common_state.current_tab == "RGB_Mueller":
			ax.axis("off")
		else:
			try:
				tick_x = np.linspace(0, 760, 5)
				tick_y = np.linspace(0, 540, 5)
				label_x = [f"{int(w * x / 760)}" for x in tick_x]
				label_y = [f"{int(h * (1 - y / 540))}" for y in tick_y]  # 아래가 0

				ax.set_xticks(tick_x)
				ax.set_yticks(tick_y)
				ax.set_xticklabels(label_x)
				ax.set_yticklabels(label_y)
				ax.tick_params(labelsize=8)
				ax.set_xlabel("X", fontsize=10)
				ax.set_ylabel("Y", fontsize=10)
			except:
				pass

		if not is_original:
			cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.05, pad=0.02)
			cbar.ax.tick_params(labelsize=8)

		canvas = FigureCanvas(fig)
		canvas.draw()
		image_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(canvas.get_width_height()[::-1] + (4,))
		image_array = image_array.astype(np.float32) / 255.0  # Normalize (0~1)

		common_state.last_figure = fig

		texture_name = "uploaded_texture"

		dpg.set_value(texture_name, image_array.flatten())
	finally:
		plt.close(fig)
