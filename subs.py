import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import dearpygui.dearpygui as dpg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
import scipy.io as spio
from scipy.interpolate import PchipInterpolator
from bisect import bisect

npy_data = None
selected_file_name = None
last_figure = None
current_tab = "Trichromatic"
selected_option = "original"
selected_direction = "X"
selectable_wavelengths = ["R", "G", "B"]
selected_wavelength = "R"
selected_stokes = "s0"
graph_option = "s0"
shown_histogram = 1
vmax = 0
vmin = 0
temp_abs_vmax = 0
input_vmax = 0
input_vmin = 0
upper_left_x = 0
upper_left_y = 0
lower_right_x = 0
lower_right_y = 0
visualizing_by_wavelength = False

def activate_visualization():
	global selected_wavelength, selected_stokes
	dpg.configure_item("polarimetric_options", enabled=True)
	dpg.configure_item("wavelength_options", enabled=True)
	update_wavelengths_visualization(selected_wavelength, selected_stokes)

def setup_fonts():
	with dpg.font_registry():
		dpg.add_font("C:/Windows/Fonts/arial.ttf", 18, tag="default_font")
	dpg.bind_font("default_font")

def update_wavelength_options():
	global current_tab, selected_wavelengths, selected_wavelength
	if current_tab == "Trichromatic":
		new_items = ["R", "G", "B"]
	else:
		new_items = [f"{w}nm" for w in np.arange(450, 651, 10)]

	selected_wavelengths = new_items
	selected_wavelength = new_items[0]
	dpg.configure_item("wavelength_options", items=new_items)
	dpg.set_value("wavelength_options", new_items[0])

def reload_visualization():
	global selected_option, selected_wavelength, vmin, vmax, input_vmin, input_vmax, visualizing_by_wavelength
	vmin = input_vmin
	vmax = input_vmax
	if not visualizing_by_wavelength:
		update_visualization(selected_option)
	else:
		update_wavelengths_visualization(selected_wavelength, selected_option)

def reset_visualization():
	global selected_option, vmin, vmax, temp_abs_vmax
	if selected_option == "s0" or selected_option == "dolp" or selected_option == "docp":
		vmin = 0
		vmax = 1
	else:
		vmin = -temp_abs_vmax
		vmax = temp_abs_vmax
	update_visualization(selected_option)

def check_valid_by_wavelength(visualizing):
	global selected_option, vmin, vmax
	if vmax <= vmin:
		return False

	elif visualizing in ["s0", "dolp", "docp", "Polarized (linear)", "Polarized (circular)", "Polarized (total)"] and (vmin < 0 or vmax > 1):
		return False

	return True

def set_selected_option(sender):
	global current_tab
	if current_tab == "Trichromatic" and sender == "original":
		update_visualization("original")
	elif current_tab == "Hyperspectral" and sender == "original":
		update_visualization("original_hyper")
	else:
		update_visualization(sender)  # Matplotlib 시각화 실행

def select_option_callback():
	global selected_wavelength, selected_stokes
	selected_wavelength = dpg.get_value("wavelength_options")
	selected_stokes = dpg.get_value("polarimetric_options")
	update_wavelengths_visualization(selected_wavelength, selected_stokes)

def graph_option_callback():
	global graph_option
	graph_option = dpg.get_value("graph_options")

def on_upper_left_x():
	global upper_left_x
	try:
		upper_left_x = float(dpg.get_value("upper_left_x"))
	except ValueError:
		print("Invalid coordinates")
		return

def on_upper_left_y():
	global upper_left_y
	try:
		upper_left_y = float(dpg.get_value("upper_left_y"))
	except ValueError:
		print("Invalid coordinates")
		return

def on_lower_right_x():
	global lower_right_x
	try:
		lower_right_x = float(dpg.get_value("lower_right_x"))
	except ValueError:
		print("Invalid coordinates")
		return

def on_lower_right_y():
	global lower_right_y
	try:
		lower_right_y = float(dpg.get_value("lower_right_y"))
	except ValueError:
		print("Invalid coordinates")
		return

def on_vmax_change():
	global vmax, input_vmax
	try:
		vmax_ = float(dpg.get_value("vmax_input"))
		input_vmax = vmax_
	except ValueError:
		vmax = 0

def on_vmin_change():
	global vmin, input_vmin
	try:
		vmin_ = float(dpg.get_value("vmin_input"))
		input_vmin = vmin_
	except ValueError:
		vmin = 0

def check_range_valid(vmax, vmin, visualizing):
	if vmax <= vmin:
		return False

	elif visualizing in ["original", "original_hyper", "polarized_linear", "polarized_circular", "polarized_total"]:
		return False

	elif visualizing in ["s0", "dolp", "docp"] and vmin < 0:
		return False

	return True

def show_histogram_callback(parameter):
	global selected_direction
	show_stokes_histogram(parameter=parameter, direction=selected_direction)

def change_direction(app_data):
	global selected_direction, shown_histogram
	selected_direction = app_data
	show_stokes_histogram(parameter=shown_histogram, direction=selected_direction)

def file_selected_callback(selected_files):
	global selected_file_name
	if not selected_files:
		return
	selected_file_path = selected_files[0]
	selected_file_name = os.path.basename(selected_file_path)

	load_npy_and_display(selected_file_path)

def open_export_dialog():
	with dpg.file_dialog(directory_selector=False, show=True, callback=export_image_callback, tag="export_dialog_id", width=800, height=400):
		dpg.add_file_extension(".png", color=(0, 200, 0, 255))

def export_image_callback(app_data):
	global last_figure
	save_path = app_data['file_path_name']
	if last_figure:
		last_figure.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Image saved to {save_path}")
	else:
		print("No image to save.")

def load_npy_and_display(file_path=None):
	global npy_data, current_tab

	if not file_path:
		print("No file selected.")
		return

	try:
		npy_data = np.load(file_path)  # 메모리 매핑 (속도 최적화)
		print(f"Loaded file: {file_path}, Shape: {npy_data.shape}")
		missing_mask = np.isnan(npy_data) | (npy_data < -1e6) | (npy_data > 1e6)
		missing_pixels = np.any(missing_mask, axis=(2, 3))  # (H, W) 크기의 마스크 생성
		npy_data[missing_pixels, :, :] = 0
		last_dim = npy_data.shape[-1]

		if last_dim == 3:
			current_tab = "Trichromatic"
			update_visualization("original")
		else:
			current_tab = "Hyperspectral"
			update_visualization("original_hyper")
		update_wavelength_options()

	except Exception as e:
		print(f"Unexpected error: {e}")

def update_wavelengths_visualization(selected_wavelengths, selected_stokes):
	global current_tab, npy_data, selected_option, selected_wavelength, vmin, vmax, visualizing_by_wavelength
	if npy_data is None:
		return

	stokes_index = {"s0": 0, "s1": 1, "s2": 2, "s3": 3, "DoLP": 4, "DoCP": 5, "AoLP": 6, "CoP": 7, "Unpolarized": 8,
					"Polarized (linear)": 9, "Polarized (circular)": 10, "Polarized (total)": 11}
	rgb_index = {"R": 0, "G": 1, "B": 2}
	wavelengths = [f"{450 + i*10}nm" for i in range(21)]
	hyper_index = {name: idx for idx, name in enumerate(wavelengths)}
	selected_option = selected_stokes
	selected_wavelength = selected_wavelengths

	if current_tab == "Trichromatic":
		index = rgb_index

	else: #Hyperspectral
		index = hyper_index

	s0 = npy_data[:, :, 0, index[selected_wavelengths]]
	s1 = npy_data[:, :, 1, index[selected_wavelengths]]
	s2 = npy_data[:, :, 2, index[selected_wavelengths]]
	s3 = npy_data[:, :, 3, index[selected_wavelengths]]

	if selected_stokes in ["s0", "s1", "s2", "s3"]:
		selected_data = npy_data[:, :, stokes_index[selected_stokes], index[selected_wavelengths]]
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
	elif selected_stokes in ["Polarized (linear)"]:
		polarized_linear = np.sqrt(s1 ** 2 + s2 ** 2)
		selected_data = polarized_linear
	elif selected_stokes in ["Polarized (circular)"]:
		polarized_circular = np.sqrt(s3 ** 2)
		selected_data = polarized_circular
	else: # Polarized (total)
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
		vmax_ = vmax
		vmin_ = vmin
	else:
		vmin_, vmax_ = (0, 1) if (selected_stokes in ["s0", "DoLP", "DoCP", "Unpolarized", "Polarized (linear)", "Polarized (circular)", "Polarized (total)"]) \
			else (-np.max(np.abs(selected_data)), np.max(np.abs(selected_data)))
	visualizing_by_wavelength = True
	generate_texture(selected_data, f"{selected_stokes} - {selected_wavelengths} channel", cmap, vmin_, vmax_)

def generate_texture(image_data, title, colormap=None, vmin=None, vmax=None, is_original=False):
	if is_original:
		display_image = cv2.resize(image_data, (760, 540), interpolation=cv2.INTER_LINEAR)

	elif image_data.ndim == 2:
		display_image = cv2.resize(image_data, (760, 540), interpolation=cv2.INTER_LINEAR)
		display_image = display_image / (np.max(display_image) + 1e-8)
	else:
		display_image = cv2.resize(np.mean(image_data, axis=2), (760, 540), interpolation=cv2.INTER_LINEAR)
		display_image = display_image / (np.max(display_image) + 1e-8)

	fig, ax = plt.subplots(figsize=(7.6, 5.4))
	if is_original:
		img = ax.imshow(display_image)  # cmap 없이 원본 색상 출력
	else:
		img = ax.imshow(display_image, cmap=colormap, interpolation="nearest", vmin=vmin, vmax=vmax)
	ax.axis("off")
	ax.set_title(title)

	if not is_original:
		cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.05, pad=0.02)
		cbar.ax.tick_params(labelsize=8)

	canvas = FigureCanvas(fig)
	canvas.draw()
	image_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(canvas.get_width_height()[::-1] + (4,))
	image_array = image_array.astype(np.float32) / 255.0  # Normalize (0~1)

	global last_figure
	last_figure = fig

	texture_name = "uploaded_texture"
	width, height = canvas.get_width_height()

	dpg.set_value(texture_name, image_array.flatten())
	dpg.set_item_width("uploaded_texture", 760)
	dpg.set_item_height("uploaded_texture", 540)

def generate_histogram(data_list, labels, colors, title, xlabel, ylabel, bins=200, value_range=(-0.3, 0.3)):
	fig, ax = plt.subplots(figsize=(7.6, 5.4))
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	for data, label, color in zip(data_list, labels, colors):
		hist, bin_edges = np.histogram(data, bins=bins, range=value_range)
		centers = (bin_edges[:-1] + bin_edges[1:]) / 2
		if title == "Stokes Distribution":
			ax.plot(centers, hist, label=label, color=color)
		elif title == "Gradient Distributions of Stokes Vector(X direction)" or title == "Gradient Distributions of Stokes Vector(Y direction)":
			ax.plot(centers, np.log(hist + 1e-10), label=label, color=color)
		elif title == "Gradient derivatives distributions of Stokes vector (X direction)" or title == "Gradient derivatives distributions of Stokes vector (Y direction)":
			ax.plot(centers, np.log(hist + 1e-10), label=label, color=color)
		elif title == "Gradient Distributions of Polarization Features (X direction)" or title == "Gradient Distributions of Polarization Features (Y direction)":
			ax.plot(centers, np.log(hist + 1e-10), label=label, color=color)

	ax.legend()
	plt.tight_layout()

	#  Matplotlib figure → DearPyGui 텍스처 변환
	canvas = FigureCanvas(fig)
	canvas.draw()

	image_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(canvas.get_width_height()[::-1] + (4,))
	image_array = image_array.astype(np.float32) / 255.0  # Normalize (0~1)

	dpg.set_value("uploaded_texture", image_array.flatten())
	plt.close(fig)

def show_stokes_histogram(parameter, direction="X"):
	global npy_data, shown_histogram
	shown_histogram = parameter
	if npy_data is None:
		return

	s0, s1, s2, s3 = npy_data[:, :, :, 0].transpose(2, 0, 1)

	dpg.configure_item("polarimetric_options", enabled=False)
	dpg.configure_item("wavelength_options", enabled=False)

	if parameter == 1: # Distribution of Stokes vector
		# 데이터 평탄화: (H, W, 4, 3) → (H*W*3, 4)
		stokes_labels = ["$s_0$", "$s_1$", "$s_2$", "$s_3$"]
		colors = ['blue', 'orange', 'green', 'red']
		data_flattened = npy_data.reshape(-1, npy_data.shape[2])
		data_list = [data_flattened[:, i] for i in range(data_flattened.shape[1])]
		generate_histogram(data_list, stokes_labels, colors, title="Stokes Distribution", xlabel="Value", ylabel="Number of Pixels")


	if parameter == 2: # Gradient distribution of Stokes vector
		stokes_labels = ["$s_0$", "$s_1$", "$s_2$", "$s_3$"]
		colors = ['blue', 'orange', 'green', 'red']
		gradients_x = []
		gradients_y = []
		for i in range(4):  # s0, s1, s2, s3
			grad_x, grad_y = np.gradient(npy_data[:, :, i, 0])  # Gradient of the first channel
			gradients_x.append(grad_x.flatten())  # Flatten the X-direction gradient
			gradients_y.append(grad_y.flatten())  # Flatten the Y-direction gradient

		if direction == "X":
			generate_histogram(gradients_x, stokes_labels, colors, title="Gradient Distributions of Stokes Vector(X direction)", xlabel="Gradient (X direction)", ylabel="Log Probability", value_range=(-3, 3))

		elif direction == "Y":
			generate_histogram(gradients_y, stokes_labels, colors, title="Gradient Distributions of Stokes Vector(Y direction)", xlabel="Gradient (Y direction)", ylabel="Log Probability", value_range=(-3, 3))


	if parameter == 3:
		# Compute gradients for Stokes derivatives
		stokes_labels = ["$s'_1$", "$s'_2$", "$s'_3$"]
		colors = ['orange', 'green', 'red']
		gradients_x = []
		gradients_y = []

		epsilon = 1e-10  # Prevent division by zero
		s1_prime = s1 / (s0 + epsilon)
		s2_prime = s2 / (s0 + epsilon)
		s3_prime = s3 / (s0 + epsilon)

		normalized_stokes = [s1_prime, s2_prime, s3_prime]

		# Compute gradients for Stokes derivatives
		for stokes_parameter in normalized_stokes:  # s1', s2', s3'
			grad_x, grad_y = np.gradient(stokes_parameter)  # Gradient of the first channel
			gradients_x.append(grad_x.flatten())  # Flatten the X-direction gradient
			gradients_y.append(grad_y.flatten())  # Flatten the Y-direction gradient

		if direction == "X":
			generate_histogram(gradients_x, stokes_labels, colors, title="Gradient derivatives distributions of Stokes vector (X direction)", xlabel="Gradient (X direction)", ylabel="Log Probability", value_range=(-3, 3))

		elif direction == "Y":
			generate_histogram(gradients_y, stokes_labels, colors, title="Gradient derivatives distributions of Stokes vector (Y direction)", xlabel="Gradient (Y direction)", ylabel="Log Probability", value_range=(-3, 3))


	if parameter == 4:
		# Compute polarization features
		dolp = np.sqrt(s1 ** 2 + s2 ** 2) / np.maximum(s0, 1e-6)
		docp = np.abs(s3) / np.maximum(s0, 1e-6)
		aolp = 0.5 * np.arctan2(s2, s1)
		cop = 0.5 * np.arctan2(s3, np.sqrt(s1 ** 2 + s2 ** 2))
		features = {
			"DoLP": dolp,
			"DoCP": docp,
			"AoLP": aolp,
			"CoP": cop
		}
		colors = ['red', 'blue', 'cyan', 'purple']

		gradients = []
		labels = []
		for label, feature in features.items():
			if direction == "X":
				grad, _ = np.gradient(feature)
			elif direction == "Y":
				_, grad = np.gradient(feature)
			gradients.append(grad.flatten())
			labels.append(label)

		generate_histogram(
			data_list=gradients,
			labels=labels,
			colors=colors,
			title=f"Gradient Distributions of Polarization Features ({direction} direction)",
			xlabel=f"Gradient ({direction} direction)",
			ylabel="Log Probability",
			bins=200,
			value_range=(-3, 3)
		)

def make_custom_seismic():
	colors = [(0, 0, 1), (0, 0, 0.5), (0, 0, 0), (0.5, 0.5, 0), (1, 1, 0)]  # Blue -> Yellow
	positions = [0.0, 0.25, 0.5, 0.75, 1.0]  # Color positions
	custom_seismic = LinearSegmentedColormap.from_list("BlueYellow", list(zip(positions, colors)))
	return custom_seismic

def visualize_original():
	s0 = npy_data[:, :, 0, :]
	s0[s0 < 0] = 1e-6
	original_image = s0 / np.max(s0)
	return generate_texture(original_image, "Original Image (sRGB)", "gray", vmin=0, vmax=1, is_original=True)

def visualize_hyper_rgb():
	ydim, xdim = npy_data.shape[:2]
	wavelengths = np.arange(450, 651, 10)  # 21개
	rgb_image = HSI2RGB(wavelengths, npy_data, ydim, xdim, d=65, threshold=0.02)
	return generate_texture(rgb_image, "RGB Approximation from Hyperspectral", is_original=True)

def visualize_s0():
	global vmin, vmax
	valid = check_range_valid(vmax, vmin, "s0")
	vmin_ = vmin if valid else 0
	vmax_ = vmax if valid else 1
	s0 = npy_data[:, :, 0, :]
	s0[s0 < 0] = 1e-6
	return generate_texture(s0, "s0: Total Intensity", "gray", vmin=vmin_, vmax=vmax_)

def visualize_s1():
	global vmin, vmax, temp_abs_vmax
	valid = check_range_valid(vmax, vmin, "s1")
	s1 = npy_data[:, :, 1, :]
	s1_resized = cv2.resize(np.mean(s1 / (np.max(s1) + 1e-8), axis=2), (760, 540))
	max_s1 = np.max(s1_resized)
	temp_abs_vmax = max_s1
	vmin_ = vmin if valid else -max_s1
	vmax_ = vmax if valid else max_s1
	return generate_texture(s1, "s1: Linear Polarization (0°/90°)", "seismic", vmin=vmin_, vmax=vmax_)

def visualize_s2():
	global vmin, vmax, temp_abs_vmax
	s2 = npy_data[:, :, 2, :]
	valid = check_range_valid(vmax, vmin, "s2")
	s2_resized = cv2.resize(np.mean(s2 / (np.max(s2) + 1e-8), axis=2), (600, 400))
	max_s2 = np.max(s2_resized)
	temp_abs_vmax = max_s2
	vmin_ = vmin if valid else -max_s2
	vmax_ = vmax if valid else max_s2
	return generate_texture(s2, "s2: Linear Polarization (45°/-45°)", "seismic", vmin=vmin_, vmax=vmax_)

def visualize_s3():
	global temp_abs_vmax, vmin, vmax
	valid = check_range_valid(vmax, vmin, "s3")
	s3 = npy_data[:, :, 3, :]
	s3_resized = cv2.resize(np.mean(s3 / (np.max(s3) + 1e-8), axis=2), (600, 400))
	max_s3 = np.max(s3_resized)
	temp_abs_vmax = max_s3
	vmin_ = vmin if valid else -max_s3
	vmax_ = vmax if valid else max_s3
	return generate_texture(s3, "s3: Circular Polarization", "seismic", vmin=vmin_, vmax=vmax_)

def visualize_dolp():
	global vmin, vmax
	valid = check_range_valid(vmax, vmin, "dolp")
	s0 = npy_data[:, :, 0, :]
	s1 = npy_data[:, :, 1, :]
	s2 = npy_data[:, :, 2, :]
	s0[s0 < 0] = 1e-6
	dolp = np.sqrt(s1 ** 2 + s2 ** 2) / s0
	vmin_ = vmin if valid else 0
	vmax_ = vmax if valid else 1
	return generate_texture(dolp, "DoLP: Degree of Linear Polarization", "gray", vmin=vmin_, vmax=vmax_)

def visualize_aolp():
	global temp_abs_vmax, vmin, vmax
	valid = check_range_valid(vmax, vmin, "aolp")
	s2 = npy_data[:, :, 2, :]
	s1 = npy_data[:, :, 1, :]
	aolp = 0.5 * np.arctan(s2 / s1)
	max_aolp = np.max(np.abs(aolp))
	temp_abs_vmax = max_aolp
	vmin_ = vmin if valid else -max_aolp
	vmax_ = vmax if valid else max_aolp
	return generate_texture(aolp, "AoLP: Angle of Linear Polarization", plt.cm.hsv, vmin=vmin_, vmax=vmax_)

def visualize_docp():
	global vmin, vmax
	valid = check_range_valid(vmax, vmin, "docp")
	s0 = npy_data[:, :, 0, :]
	s3 = npy_data[:, :, 3, :]
	s0[s0 < 0] = 1e-6
	docp = np.abs(s3) / np.maximum(s0, 1e-6)
	vmin_ = vmin if valid else 0
	vmax_ = vmax if valid else 1
	return generate_texture(docp, "DoCP: Degree of Circular Polarization", "gray", vmin=vmin_, vmax=vmax_)

def visualize_cop():
	global vmin, vmax, temp_abs_vmax
	valid = check_range_valid(vmax, vmin, "aolp")
	s2 = npy_data[:, :, 2, :]
	s1 = npy_data[:, :, 1, :]
	s3 = npy_data[:, :, 3, :]
	cop = 0.5 * np.arctan(s3 / np.sqrt(s1**2 + s2**2))
	max_cop = np.max(np.abs(cop))
	temp_abs_vmax = max_cop
	vmin_ = vmin if valid else -max_cop
	vmax_ = vmax if valid else max_cop
	custom_seismic = make_custom_seismic()
	return generate_texture(cop, "CoP: Chirality of Polarization", custom_seismic, vmin=vmin_, vmax=vmax_)

def visualize_unpolarized():
	s0 = npy_data[:, :, 0, :]
	s2 = npy_data[:, :, 2, :]
	s1 = npy_data[:, :, 1, :]
	s3 = npy_data[:, :, 3, :]
	unpolarized = s0 - np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
	return generate_texture(unpolarized, "Unpolarized", is_original=True)

def visualize_linear_polarized():
	s2 = npy_data[:, :, 2, :]
	s1 = npy_data[:, :, 1, :]
	polarized = np.sqrt(s1 ** 2 + s2 ** 2)
	polarized = np.clip(polarized, 0, 1)
	return generate_texture(polarized, "Polarized (linear)", is_original=True)

def visualize_circular_polarized():
	s3 = npy_data[:, :, 3, :]
	polarized = np.sqrt(s3 ** 2)
	polarized = np.clip(polarized, 0, 1)
	return generate_texture(polarized, "Polarized (circular)", is_original=True)

def visualize_total_polarized():
	s2 = npy_data[:, :, 2, :]
	s1 = npy_data[:, :, 1, :]
	s3 = npy_data[:, :, 3, :]
	polarized = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
	polarized = np.clip(polarized, 0, 1)
	return generate_texture(polarized, "Polarized (total)", is_original=True)

# 선택한 옵션에 따라 함수 실행
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

def update_visualization(option):
	global npy_data, selected_option, visualizing_by_wavelength
	if npy_data is None:
		return
	selected_option = option
	dpg.configure_item("polarimetric_options", enabled=False)
	dpg.configure_item("wavelength_options", enabled=False)
	# 선택한 옵션에 맞는 시각화 함수 실행
	if option in visualization_functions:
		visualizing_by_wavelength = False
		visualization_functions[option]()  # 해당 함수 실행
	else:
		print(f"Invalid option: {option}")

def view_graph():
	global npy_data, upper_left_x, upper_left_y, lower_right_x, lower_right_y, graph_option

	if npy_data is None:
		return

	try:
		x1, x2 = sorted([int(upper_left_x), int(lower_right_x)])
		y1, y2 = sorted([int(upper_left_y), int(lower_right_y)])
		height, width = npy_data.shape[:2]
		x1, x2 = max(0, x1), min(width, x2)
		y1, y2 = max(0, y1), min(height, y2)

		fig, ax = plt.subplots()
		s0_crop = npy_data[y1:y2, x1:x2, 0, :]  # s0
		s1_crop = npy_data[y1:y2, x1:x2, 1, :]  # s0
		s2_crop = npy_data[y1:y2, x1:x2, 2, :]  # s0
		s3_crop = npy_data[y1:y2, x1:x2, 3, :]  # s0

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

		dolp_crop = np.sqrt(s1_crop**2 + s2_crop**2) / s0_crop
		aolp_crop = 0.5 * np.arctan2(s2_crop, s1_crop)
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

		if npy_data.ndim == 4 and npy_data.shape[2] == 4:
			selected_crop = data_map.get(graph_option)
			mean_values = np.mean(selected_crop, axis=(0, 1))

			band_count = npy_data.shape[3]
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

			ax.set_title(f"Mean {graph_option} across wavelengths")
			ax.set_ylabel(f"Mean {graph_option}")
			ax.grid(True)

			canvas = FigureCanvas(fig)
			canvas.draw()
			image_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(
				canvas.get_width_height()[::-1] + (4,))
			image_array = image_array.astype(np.float32) / 255.0
			plt.close(fig)

			if not dpg.does_item_exist("graph_texture"):
				with dpg.texture_registry(show=False):
					dpg.add_dynamic_texture(width=image_array.shape[1], height=image_array.shape[0],
											default_value=image_array.flatten(), tag="graph_texture")
			else:
				dpg.set_value("graph_texture", image_array.flatten())

			if not dpg.does_item_exist("graph_window"):
				with dpg.window(label="Graph Window", tag="graph_window", width=700, height=500, pos=(100, 100)):
					dpg.add_image("graph_texture")
			else:
				dpg.configure_item("graph_window", show=True)

		else:
			print("Unsupported data format.")
			return

		plt.show()

	except Exception as e:
		print(f"[ERROR] Failed to plot region summary: {e}")


def HSI2RGB(wY, stokes_data, ydim, xdim, d, threshold):
	# Load reference illuminant
	D = spio.loadmat('C:/2024 Winter Internship/SP_data_visualization/D_illuminants.mat')
	w = D['wxyz'][:, 0]
	x = D['wxyz'][:, 1]
	y = D['wxyz'][:, 2]
	z = D['wxyz'][:, 3]
	D = D['D']

	i = {50: 2, 55: 3, 65: 1, 75: 4}
	wI = D[:, 0]
	I = D[:, i[d]]

	# Interpolate to match image wavelengths
	I = PchipInterpolator(wI, I, extrapolate=True)(wY)
	x = PchipInterpolator(w, x, extrapolate=True)(wY)
	y = PchipInterpolator(w, y, extrapolate=True)(wY)
	z = PchipInterpolator(w, z, extrapolate=True)(wY)

	# Truncate at 780nm
	i = bisect(wY, 780)
	wY = wY[:i]
	I = I[:i]
	x = x[:i]
	y = y[:i]
	z = z[:i]

	# Extract s0 (Total Intensity)
	s0 = stokes_data[:, :, 0, :]  # (ydim, xdim, 21)

	# Normalize s0
	s0 = s0 / np.max(s0)

	# Compute k (Scaling factor)
	k = 1 / np.trapezoid(y * I, wY)

	# Compute X, Y, Z for image
	X = k * np.trapezoid(s0 * (I * x), wY, axis=2)
	Y = k * np.trapezoid(s0 * (I * y), wY, axis=2)
	Z = k * np.trapezoid(s0 * (I * z), wY, axis=2)

	XYZ = np.stack([X, Y, Z], axis=-1)  # Shape: (ydim, xdim, 3)

	# Convert XYZ to RGB using sRGB matrix
	M = np.array([[3.2404542, -1.5371385, -0.4985314],
				  [-0.9692660, 1.8760108, 0.0415560],
				  [0.0556434, -0.2040259, 1.0572252]])

	sRGB = np.dot(XYZ, M.T)

	# Gamma correction
	gamma_map = sRGB > 0.0031308
	sRGB[gamma_map] = 1.055 * np.power(sRGB[gamma_map], (1. / 2.4)) - 0.055
	sRGB[~gamma_map] = 12.92 * sRGB[~gamma_map]

	# Clip values to valid range
	sRGB = np.clip(sRGB, 0, 1)

	# Apply thresholding (contrast adjustment)
	if threshold:
		for idx in range(3):
			y = sRGB[:, :, idx]
			a, b = np.histogram(y, 100)
			b = b[:-1] + np.diff(b) / 2
			a = np.cumsum(a) / np.sum(a)
			th = b[0]
			i = a < threshold
			if i.any():
				th = b[i][-1]
			y = y - th
			y[y < 0] = 0

			a, b = np.histogram(y, 100)
			b = b[:-1] + np.diff(b) / 2
			a = np.cumsum(a) / np.sum(a)
			i = a > 1 - threshold
			th = b[i][0]
			y[y > th] = th
			y = y / th
			sRGB[:, :, idx] = y

	return sRGB