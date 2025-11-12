from subs.SP_image import sp_state
from subs import common_state
import dearpygui.dearpygui as dpg
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle
from .hsi_conversion import HSI2RGB
from subs.themes import make_custom_seismic
matplotlib.use('Agg')
wavelengths = np.arange(450, 651, 10)

def check_valid_by_wavelength(visualizing):
	if common_state.vmax <= common_state.vmin:
		return False

	elif (visualizing in ["s0", "dolp", "docp", "Polarized (linear)", "Polarized (circular)", "Polarized (total)"]
		  and (common_state.vmin < 0 or common_state.vmax > 1)):
		return False

	return True

def check_range_valid(vmax, vmin, visualizing):
	if vmax <= vmin:
		return False

	elif visualizing in ["original", "original_hyper", "polarized_linear", "polarized_circular", "polarized_total"]:
		return False

	elif visualizing in ["s0", "DoLP", "DoCP"] and vmin < 0:
		return False

	return True

def update_visualization(option):
	if common_state.npy_data is None:
		return
	common_state.selected_option = option
	dpg.configure_item("polarimetric_options", enabled=False)
	dpg.configure_item("wavelength_options", enabled=False)

	if (option in visualization_functions) and common_state.current_tab != "RGB_Mueller":
		sp_state.visualizing_by_wavelength = False
		visualization_functions[option]()
	else:
		return

def update_wavelengths_visualization(selected_wavelengths, selected_stokes):
	if common_state.npy_data is None:
		return

	stokes_index = {"s0": 0, "s1": 1, "s2": 2, "s3": 3, "DoLP": 4, "DoCP": 5, "AoLP": 6, "CoP": 7, "Unpolarized": 8,
					"Polarized(Linear)": 9, "Polarized(Circular)": 10, "Polarized(total)": 11}
	rgb_index = {"R": 0, "G": 1, "B": 2}
	hyper_wavelengths = [f"{450 + i*10}nm" for i in range(21)]
	hyper_index = {name: idx for idx, name in enumerate(hyper_wavelengths)}
	sp_state.selected_option = selected_stokes
	sp_state.selected_wavelength = selected_wavelengths

	if common_state.current_tab == "Trichromatic":
		index = rgb_index

	else: #Hyperspectral
		index = hyper_index

	s0 = common_state.npy_data[:, :, 0, index[selected_wavelengths]]
	s1 = common_state.npy_data[:, :, 1, index[selected_wavelengths]]
	s2 = common_state.npy_data[:, :, 2, index[selected_wavelengths]]
	s3 = common_state.npy_data[:, :, 3, index[selected_wavelengths]]

	if selected_stokes in ["s0", "s1", "s2", "s3"]:
		selected_data = common_state.npy_data[:, :, stokes_index[selected_stokes], index[selected_wavelengths]]
	elif selected_stokes in ["DoLP"]:
		dolp = np.sqrt(s1 ** 2 + s2 ** 2) / s0
		selected_data = dolp
	elif selected_stokes in ["DoCP"]:
		docp = np.abs(s3) / s0
		selected_data = docp
	elif selected_stokes in ["AoLP"]:
		aolp = 0.5 * np.arctan2(s2, s1)
		selected_data = aolp
	elif selected_stokes in ["CoP"]:
		cop = 0.5 * np.arctan2(s3, np.sqrt(s1 ** 2 + s2 ** 2))
		selected_data = cop
	elif selected_stokes in ["Unpolarized"]:
		unpolarized = s0 - np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
		selected_data = unpolarized
	elif selected_stokes in ["Polarized(Linear)"]:
		polarized_linear = np.sqrt(s1 ** 2 + s2 ** 2)
		selected_data = polarized_linear
	elif selected_stokes in ["Polarized(Circular)"]:
		polarized_circular = np.sqrt(s3 ** 2)
		selected_data = polarized_circular
	else: # Polarized(total)
		polarized_total = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
		selected_data = polarized_total

	# 시각화 범위 설정
	if selected_stokes in ["s1", "s2", "s3"]:
		cmap = "seismic"
	elif selected_stokes in ["CoP"]:
		custom_seismic = make_custom_seismic()
		cmap = custom_seismic
	elif selected_stokes in ["AoLP"]:
		cmap = "hsv"
	else:
		cmap = "gray"

	valid = check_valid_by_wavelength(selected_stokes)
	if valid:
		vmax_ = common_state.vmax
		vmin_ = common_state.vmin
	else:
		vmin_, vmax_ = (0, 1) if (selected_stokes in ["s0", "DoLP", "DoCP", "Unpolarized", "Polarized(Linear)", "Polarized(Circular)", "Polarized(total)"]) \
			else (-np.max(np.abs(selected_data)), np.max(np.abs(selected_data)))
	sp_state.visualizing_by_wavelength = True
	generate_texture(selected_data, f"{selected_stokes} - {selected_wavelengths} channel", cmap, vmin_, vmax_)

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

		# Draw rectangle overlay (Graph region)
		if sp_state.show_rectangle_overlay and (sp_state.upper_right_x > sp_state.lower_left_x) and (
				sp_state.upper_right_y > sp_state.lower_left_y):
			x_scale = 760 / w
			y_scale = 540 / h

			rect_x = sp_state.lower_left_x * x_scale
			rect_y = (h - sp_state.upper_right_y) * y_scale
			rect_w = (sp_state.upper_right_x - sp_state.lower_left_x) * x_scale
			rect_h = (sp_state.upper_right_y - sp_state.lower_left_y) * y_scale

			# Rectangle(left, bottom), width, height
			rect = Rectangle(
				(rect_x, rect_y), rect_w, rect_h,
				linewidth=2.0,
				edgecolor='red',
				facecolor='red',
				alpha=0.3
			)
			ax.add_patch(rect)

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
		dpg.set_item_width("uploaded_texture", 760)
		dpg.set_item_height("uploaded_texture", 540)
	finally:
		plt.close(fig)

def visualize_original():
	s0 = common_state.npy_data[:, :, 0, :]
	s0[s0 < 0] = 1e-6
	original_image = s0 / np.max(s0)
	return generate_texture(original_image, "Original Image (sRGB)", "gray", vmin=0, vmax=1, is_original=True)

def visualize_hyper_rgb():
	s0 = common_state.npy_data[:, :, 0, :]
	rgb_image = HSI2RGB(wavelengths, s0, d=65, threshold=0.02)
	return generate_texture(rgb_image, "RGB Approximation from Hyperspectral", is_original=True)

def visualize_s0():
	valid = check_range_valid(common_state.vmax, common_state.vmin, "s0")
	s0 = common_state.npy_data[:, :, 0, :]
	s0[s0 < 0] = 1e-6
	vmin_ = common_state.vmin if valid else 0
	vmax_ = common_state.vmax if valid else np.max(s0)
	return generate_texture(s0, "s0: Total Intensity", "gray", vmin=vmin_, vmax=vmax_)

def visualize_s1():
	valid = check_range_valid(common_state.vmax, common_state.vmin, "s1")
	s1 = common_state.npy_data[:, :, 1, :]
	s1_resized = cv2.resize(np.mean(s1 / (np.max(s1) + 1e-8), axis=2), (760, 540))
	max_s1 = np.max(s1_resized)
	common_state.temp_abs_vmax = max_s1
	vmin_ = common_state.vmin if valid else -max_s1
	vmax_ = common_state.vmax if valid else max_s1
	return generate_texture(s1, "s1: Linear Polarization (0°/90°)", "seismic", vmin=vmin_, vmax=vmax_, normalize=False)

def visualize_s2():
	s2 = common_state.npy_data[:, :, 2, :]
	valid = check_range_valid(common_state.vmax, common_state.vmin, "s2")
	s2_resized = cv2.resize(np.mean(s2 / (np.max(s2) + 1e-8), axis=2), (760, 540))
	max_s2 = np.max(s2_resized)
	common_state.temp_abs_vmax = max_s2
	vmin_ = common_state.vmin if valid else -max_s2
	vmax_ = common_state.vmax if valid else max_s2
	return generate_texture(s2, "s2: Linear Polarization (45°/-45°)", "seismic", vmin=vmin_, vmax=vmax_, normalize=False)

def visualize_s3():
	valid = check_range_valid(common_state.vmax, common_state.vmin, "s3")
	s3 = common_state.npy_data[:, :, 3, :]
	s3_resized = cv2.resize(np.mean(s3 / (np.max(s3) + 1e-8), axis=2), (760, 540))
	max_s3 = np.max(s3_resized)
	common_state.temp_abs_vmax = max_s3
	vmin_ = common_state.vmin if valid else -max_s3
	vmax_ = common_state.vmax if valid else max_s3
	return generate_texture(s3, "s3: Circular Polarization", "seismic", vmin=vmin_, vmax=vmax_, normalize=False)

def visualize_dolp():
	valid = check_range_valid(common_state.vmax, common_state.vmin, "dolp")
	s0 = common_state.npy_data[:, :, 0, :]
	s1 = common_state.npy_data[:, :, 1, :]
	s2 = common_state.npy_data[:, :, 2, :]
	s0[s0 <= 0] = 1e-6
	dolp = np.sqrt(s1 ** 2 + s2 ** 2) / s0
	vmin_ = common_state.vmin if valid else 0
	vmax_ = common_state.vmax if valid else 1
	return generate_texture(dolp, "DoLP: Degree of Linear Polarization", "gray", vmin=vmin_, vmax=vmax_)

def visualize_aolp():
	valid = check_range_valid(common_state.vmax, common_state.vmin, "aolp")
	s2 = common_state.npy_data[:, :, 2, :]
	s1 = common_state.npy_data[:, :, 1, :]
	aolp = 0.5 * np.arctan2(s2, s1)
	max_aolp = np.max(np.abs(aolp))
	common_state.temp_abs_vmax = max_aolp
	vmin_ = common_state.vmin if valid else -max_aolp
	vmax_ = common_state.vmax if valid else max_aolp
	return generate_texture(aolp, "AoLP: Angle of Linear Polarization", plt.cm.hsv, vmin=vmin_, vmax=vmax_)

def visualize_docp():
	valid = check_range_valid(common_state.vmax, common_state.vmin, "docp")
	s0 = common_state.npy_data[:, :, 0, :]
	s3 = common_state.npy_data[:, :, 3, :]
	s0[s0 < 0] = 1e-6
	docp = np.abs(s3) / np.maximum(s0, 1e-6)
	vmin_ = common_state.vmin if valid else 0
	vmax_ = common_state.vmax if valid else 1
	return generate_texture(docp, "DoCP: Degree of Circular Polarization", "gray", vmin=vmin_, vmax=vmax_)

def visualize_cop():
	valid = check_range_valid(common_state.vmax, common_state.vmin, "aolp")
	s2 = common_state.npy_data[:, :, 2, :]
	s1 = common_state.npy_data[:, :, 1, :]
	s3 = common_state.npy_data[:, :, 3, :]
	cop = 0.5 * np.arctan(s3 / np.sqrt(s1**2 + s2**2))
	max_cop = np.max(np.abs(cop))
	common_state.temp_abs_vmax = max_cop
	vmin_ = common_state.vmin if valid else -max_cop
	vmax_ = common_state.vmax if valid else max_cop
	custom_seismic = make_custom_seismic()
	return generate_texture(cop, "CoP: Chirality of Polarization", custom_seismic, vmin=vmin_, vmax=vmax_)

def visualize_unpolarized():
	s0 = common_state.npy_data[:, :, 0, :]
	s1 = common_state.npy_data[:, :, 1, :]
	s2 = common_state.npy_data[:, :, 2, :]
	s3 = common_state.npy_data[:, :, 3, :]
	unpolarized = s0 - np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
	if common_state.current_tab == "Trichromatic":
		return generate_texture(unpolarized, "Unpolarized", is_original=True)
	else:
		unpolarized_hyper = HSI2RGB(wavelengths, unpolarized, d=65, threshold=0.02)
		return generate_texture(unpolarized_hyper, "Unpolarized", is_original=True)

def visualize_linear_polarized():
	s2 = common_state.npy_data[:, :, 2, :]
	s1 = common_state.npy_data[:, :, 1, :]
	polarized = np.sqrt(s1 ** 2 + s2 ** 2)
	if common_state.current_tab == "Trichromatic":
		return generate_texture(polarized, "Polarized (linear)", is_original=True)
	else:
		polarized_hyper = HSI2RGB(wavelengths, polarized, d=65, threshold=0.02)
		return generate_texture(polarized_hyper, "Polarized (linear)", is_original=True)

def visualize_circular_polarized():
	s3 = common_state.npy_data[:, :, 3, :]
	polarized = np.sqrt(s3 ** 2)
	if common_state.current_tab == "Trichromatic":
		return generate_texture(polarized, "Polarized (circular)", is_original=True)
	else:
		polarized_hyper = HSI2RGB(wavelengths, polarized, d=65, threshold=0.02)
		return generate_texture(polarized_hyper, "Polarized (circular)", is_original=True)

def visualize_total_polarized():
	s2 = common_state.npy_data[:, :, 2, :]
	s1 = common_state.npy_data[:, :, 1, :]
	s3 = common_state.npy_data[:, :, 3, :]
	polarized = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
	if common_state.current_tab == "Trichromatic":
		return generate_texture(polarized, "Polarized (total)", is_original=True)
	else:
		polarized_hyper = HSI2RGB(wavelengths, polarized, d=65, threshold=0.02)
		return generate_texture(polarized_hyper, "Polarized (total)", is_original=True)

visualization_functions = {
	"original": visualize_original,
	"original_hyper" : visualize_hyper_rgb,
	"s0" : visualize_s0,
	"s1": visualize_s1,
	"s2": visualize_s2,
	"s3": visualize_s3,
	"DoLP": visualize_dolp,
	"AoLP": visualize_aolp,
	"DoCP": visualize_docp,
	"CoP": visualize_cop,
	"Unpolarized": visualize_unpolarized,
	"Polarized (linear)": visualize_linear_polarized,
	"Polarized (circular)" : visualize_circular_polarized,
	"Polarized (total)" : visualize_total_polarized
}

