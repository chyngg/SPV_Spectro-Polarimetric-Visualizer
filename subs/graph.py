from . import state
from .visualization import update_visualization
from .callbacks import on_close_graph_window
import numpy as np
import matplotlib.pyplot as plt
import dearpygui.dearpygui as dpg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def view_graph():
	if state.npy_data is None:
		return

	try:
		state.show_rectangle_overlay = True
		x1, x2 = sorted([int(state.upper_left_x), int(state.lower_right_x)])
		y1, y2 = sorted([int(state.upper_left_y), int(state.lower_right_y)])
		height, width = state.npy_data.shape[:2]
		x1, x2 = max(0, x1), min(width, x2)
		y1, y2 = max(0, y1), min(height, y2)

		fig, ax = plt.subplots()
		s0_crop = state.npy_data[y1:y2, x1:x2, 0, :]  # s0
		s1_crop = state.npy_data[y1:y2, x1:x2, 1, :]  # s0
		s2_crop = state.npy_data[y1:y2, x1:x2, 2, :]  # s0
		s3_crop = state.npy_data[y1:y2, x1:x2, 3, :]  # s0

		def normalize(x):
			min_val = np.min(x)
			max_val = np.max(x)
			if max_val - min_val == 0:
				return np.zeros_like(x)
			return (x - min_val) / (max_val - min_val)

		s0_crop = normalize(s0_crop)
		s1_crop = normalize(s1_crop)
		s2_crop = normalize(s2_crop)
		s3_crop = normalize(s3_crop)

		dolp_crop = np.sqrt(s1_crop**2 + s2_crop**2) /np.maximum(s0_crop, 1e-6)
		aolp_crop = 0.5 * np.arctan2(s2_crop, np.maximum(s1_crop, 1e-6))
		docp_crop = np.abs(s3_crop) / np.maximum(s0_crop, 1e-6)
		cop_crop = 0.5 * np.arctan2(s3_crop, np.sqrt(s1_crop**2 + s2_crop**2))
		data_map = {
			"s0": s0_crop,
			"s1": s1_crop,
			"s2": s2_crop,
			"s3": s3_crop,
			"DoLP": dolp_crop,
			"AoLP": aolp_crop,
			"DoCP": docp_crop,
			"CoP": cop_crop
		}

		if state.npy_data.ndim == 4 and state.npy_data.shape[2] == 4:
			selected_crop = data_map.get(state.graph_option)
			mean_values = np.mean(selected_crop, axis=(0, 1))

			band_count = state.npy_data.shape[3]
			ax.set_xticks(np.arange(450, 651, 20))  # 20nm 간격
			if band_count == 21: # Hyperspectral
				wavelengths = np.arange(450, 651, 10)  # Hyperspectral: 450~650nm
				ax.set_xlabel("Wavelength (nm)")
				ax.plot(wavelengths, mean_values, marker='o')
			elif band_count == 3: # Trichromatic
				wavelengths = [460, 540, 620]  # Trichromatic: blue, green, red
				ax.set_xlabel("Channel")
				rgb_labels = ["Blue (460nm)", "Green (540nm)", "Red (620nm)"]
				ax.set_xlabel("Channel")
				ax.bar(rgb_labels, mean_values, color=["blue", "green", "red"])

			ax.set_title(f"Mean {state.graph_option} across wavelengths")
			ax.set_ylabel(f"Mean {state.graph_option}")
			ax.grid(True)

			canvas = FigureCanvas(fig)
			canvas.draw()
			image_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(
				canvas.get_width_height()[::-1] + (4,))
			image_array = image_array.astype(np.float32) / 255.0
			update_visualization(state.selected_option)

			if not dpg.does_item_exist("graph_texture"):
				with dpg.texture_registry(show=False):
					dpg.add_dynamic_texture(width=image_array.shape[1], height=image_array.shape[0],
											default_value=image_array.flatten(), tag="graph_texture")
			else:
				dpg.set_value("graph_texture", image_array.flatten())

			if not dpg.does_item_exist("graph_window"):
				with dpg.window(label="Graph Window", tag="graph_window", width=700, height=500, pos=(100, 100), on_close=on_close_graph_window):
					dpg.add_image("graph_texture")
			else:
				dpg.configure_item("graph_window", show=True)


		else:
			print("Unsupported data format.")
			return

	except Exception as e:
		print(f"[ERROR] Failed to plot region summary: {e}")

	finally:
		plt.close()