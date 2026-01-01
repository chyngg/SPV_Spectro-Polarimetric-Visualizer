from subs.SP_image import sp_state
from subs.Mueller_matrix_image import mueller_state, mueller_video
from subs import common_state
import dearpygui.dearpygui as dpg
from subs.SP_image.sp_visualization import update_visualization, update_wavelengths_visualization
from subs.Mueller_matrix_image.mueller_visualization import visualize_rgb_mueller_grid, visualize_rgb_mueller_rgbgrid
from subs.SP_image.histogram import show_stokes_histogram
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

CANVAS_WIDTH = 760
CANVAS_HEIGHT = 540

SPLIT_DISPLAY_WIDTH = CANVAS_WIDTH // 2
SPLIT_DISPLAY_HEIGHT = int(SPLIT_DISPLAY_WIDTH * CANVAS_HEIGHT / CANVAS_WIDTH)
VERTICAL_PADDING = (CANVAS_HEIGHT - SPLIT_DISPLAY_HEIGHT) // 2

mueller_elements = [f"m{i}{j}" for i in range(4) for j in range(4)]
mueller_hist_selected = {"m00"}
mueller_hist_open = False

def change_direction(sender, app_data):
	sp_state.selected_direction = app_data
	show_stokes_histogram(parameter=sp_state.shown_histogram, direction=sp_state.selected_direction)

def show_histogram_callback(parameter):
	if common_state.current_tab == "Mueller_image":
		return
	show_stokes_histogram(parameter=parameter, direction=sp_state.selected_direction)

def set_sp_option(sender):
	if common_state.current_tab == "Trichromatic" and sender == "original":
		sp_state.sp_visualizing = "original"
	elif common_state.current_tab == "Hyperspectral" and sender == "original":
		sp_state.sp_visualizing = "original_hyper"
	else:
		sp_state.sp_visualizing = sender
	update_visualization(sp_state.sp_visualizing)

def set_sp_by_channel_callback():
	sp_state.selected_wavelength = dpg.get_value("wavelength_options")
	sp_state.selected_stokes = dpg.get_value("polarimetric_options")
	update_wavelengths_visualization(sp_state.selected_wavelength, sp_state.selected_stokes)

def change_polarized_option_callback(sender):
	sp_state.now_polarized = dpg.get_value("select_polarized_option")
	if sp_state.sp_visualizing in ["Polarized(Linear)", "Polarized(Circular)", "Polarized(total)"]:
		sp_state.sp_visualizing = f"Polarized({sp_state.now_polarized})"
		update_visualization(sp_state.sp_visualizing)

def activate_visualization():
	if common_state.current_tab == "Mueller_image":
		return
	dpg.configure_item("polarimetric_options", enabled=True)
	dpg.configure_item("wavelength_options", enabled=True)
	update_wavelengths_visualization(sp_state.selected_wavelength, sp_state.selected_stokes)

def reload_visualization():
	(common_state.vmin, common_state.vmax) = (common_state.input_vmin, common_state.input_vmax)
	if sp_state.visualizing_by_wavelength:
		update_wavelengths_visualization(sp_state.selected_wavelength, common_state.selected_option)
	elif common_state.current_tab != "Mueller_image" and common_state.current_tab != "Mueller_video":
		update_visualization(common_state.selected_option)
	elif common_state.current_tab == "Mueller_image": #Mueller_image
		if common_state.vmin < common_state.vmax:
			current_channel = common_state.wavelength_options[mueller_state.mueller_selected_channel]
			visualize_rgb_mueller_grid(common_state.npy_data, channel=current_channel, vmin=common_state.vmin, vmax=common_state.vmax)
	else: # Mueller_video
		mueller_video.on_mode_or_channel_changed()

def reset_visualization():
	if common_state.current_tab == "Mueller_image":
		current_channel = common_state.wavelength_options[mueller_state.mueller_selected_channel]
		(common_state.vmin, common_state.vmax) = (-1, 1)
		visualize_rgb_mueller_grid(common_state.npy_data, channel=current_channel, vmin=-1, vmax=1)
	else:
		if common_state.selected_option == "s0" or common_state.selected_option == "DoLP" or common_state.selected_option == "DoCP":
			(common_state.vmin, common_state.vmax) = (0, 1)
		else:
			(common_state.vmin, common_state.vmax) = (-common_state.temp_abs_vmax, common_state.temp_abs_vmax)
		update_visualization(common_state.selected_option)

def crop_graph_option_callback():
	sp_state.crop_graph_option = dpg.get_value("crop_graph_options")

def multi_graph_option_callback():
	sp_state.multi_graph_option = dpg.get_value("multi_graph_options")

def on_close_graph_window():
	sp_state.show_rectangle_overlay = False
	update_visualization(common_state.selected_option)

def on_lower_left_x():
	try:
		sp_state.lower_left_x = float(dpg.get_value("lower_left_x"))
	except ValueError:
		return

def on_lower_left_y():
	try:
		sp_state.lower_left_y = float(dpg.get_value("lower_left_y"))
	except ValueError:
		return

def on_upper_right_x():
	try:
		sp_state.upper_right_x = float(dpg.get_value("upper_right_x"))
	except ValueError:
		return

def on_upper_right_y():
	try:
		sp_state.upper_right_y = float(dpg.get_value("upper_right_y"))
	except ValueError:
		return

def on_vmax_change():
	try:
		vmax_ = float(dpg.get_value("vmax_input"))
		common_state.input_vmax = vmax_
	except ValueError:
		common_state.vmax = 0

def on_vmin_change():
	try:
		vmin_ = float(dpg.get_value("vmin_input"))
		common_state.input_vmin = vmin_
	except ValueError:
		common_state.vmin = 0

# ---- For mueller-matrix visualization ----

def mueller_select_option_callback():
	mueller_state.mueller_selected_correction = dpg.get_value("mueller_correction")

	if mueller_state.mueller_visualizing in ["Original", "m00", "Gamma"]:
		mueller_state.mueller_visualizing = mueller_state.mueller_selected_correction

	if mueller_state.is_video:
		mueller_video.on_mode_or_channel_changed()
		return

	if mueller_state.mueller_visualizing in ["Original", "m00", "Gamma"]:
		visualize_rgb_mueller_grid(
			common_state.npy_data,
			channel=mueller_state.mueller_selected_channel,
			correction=mueller_state.mueller_selected_correction,
			vmin=-1,
			vmax=1,
		)
	else:
		visualize_rgb_mueller_rgbgrid(
			common_state.npy_data,
			mueller_state.mueller_selected_correction,
			sign=mueller_state.mueller_visualizing,
		)

def mueller_channel_callback():
	mueller_state.mueller_selected_channel = dpg.get_value("mueller_channel")
	mueller_state.mueller_visualizing = mueller_state.mueller_selected_correction

	if mueller_state.is_video:
		mueller_video.on_mode_or_channel_changed()
	else:
		mueller_select_option_callback()

def mueller_rgb_callback_positive():
	mueller_state.mueller_visualizing = "Positive"
	if mueller_state.is_video:
		mueller_video.on_mode_or_channel_changed()
	else:
		visualize_rgb_mueller_rgbgrid(
			common_state.npy_data,
			mueller_state.mueller_selected_correction,
			sign="Positive",
		)

def mueller_rgb_callback_negative():
	mueller_state.mueller_visualizing = "Negative"
	if mueller_state.is_video:
		mueller_video.on_mode_or_channel_changed()
	else:
		visualize_rgb_mueller_rgbgrid(
			common_state.npy_data,
			mueller_state.mueller_selected_correction,
			sign="Negative",
		)

def on_gamma_change():
	mueller_state.gamma = float(dpg.get_value("gamma_input"))
	if not mueller_state.visualizing_gamma:
		return

	if mueller_state.is_video:
		mueller_video.on_mode_or_channel_changed()
	else:
		if mueller_state.mueller_visualizing in mueller_state.mueller_rgb_options:
			visualize_rgb_mueller_rgbgrid(
				common_state.npy_data,
				mueller_state.mueller_selected_correction,
				sign=mueller_state.mueller_visualizing,
			)
		else:
			visualize_rgb_mueller_grid(
				common_state.npy_data,
				channel=mueller_state.mueller_selected_channel,
				correction=mueller_state.mueller_selected_correction,
				vmin=-1,
				vmax=1,
			)

# ---- For Mueller histogram ----

def mueller_hist_checkbox_callback(sender, app_data):
	label = dpg.get_item_label(sender)
	if label not in mueller_elements:
		return

	if app_data:
		mueller_hist_selected.add(label)
	else:
		mueller_hist_selected.discard(label)

def enable_mueller_hist_side_by_side():
	if not dpg.does_item_exist("center_split_container"):
		return

	dpg.delete_item("center_split_container", children_only=True)

	with dpg.group(parent="center_split_container",
				   tag="center_split_row",
				   horizontal=True):

		with dpg.child_window(tag="center_left_panel",
							  width=CANVAS_WIDTH // 2,
							  height=CANVAS_HEIGHT,
							  no_scrollbar=True):
			if VERTICAL_PADDING > 0:
				dpg.add_spacer(height=VERTICAL_PADDING)
			dpg.add_image(
				"uploaded_texture",
				width=SPLIT_DISPLAY_WIDTH,
				height=SPLIT_DISPLAY_HEIGHT,
			)

		with dpg.child_window(tag="center_right_panel",
							  width=CANVAS_WIDTH // 2,
							  height=CANVAS_HEIGHT,
							  no_scrollbar=True):
			if VERTICAL_PADDING > 0:
				dpg.add_spacer(height=VERTICAL_PADDING)
			dpg.add_image(
				"histogram_texture",
				width=SPLIT_DISPLAY_WIDTH,
				height=SPLIT_DISPLAY_HEIGHT,
			)

def update_mueller_hist_if_open():
	global mueller_hist_open
	if not mueller_hist_open:
		return
	render_mueller_histogram_to_texture("histogram_texture")

def render_mueller_histogram_to_texture(texture_tag: str):
	npy = common_state.npy_data
	if npy is None:
		return

	if npy.ndim == 4 and npy.shape[-2:] == (4, 4):
		mats = npy  # (H, W, 4, 4)
		ch = getattr(mueller_state, "mueller_selected_channel", "R")
	elif npy.ndim == 5 and npy.shape[2:] == (3, 4, 4):
		ch = mueller_state.mueller_selected_channel  # "R","G","B"
		ch_idx_map = {"B": 0, "G": 1, "R": 2}
		ch_idx = ch_idx_map.get(ch, 2)  # default R
		mats = npy[:, :, ch_idx, :, :]  # (H,W,4,4)
	elif npy.ndim == 6 and npy.shape[-3:] == (3, 4, 4):
		try:
			t = int(mueller_video.player.t)
		except Exception:
			t = 0
		frame = npy[t]  # (H,W,3,4,4)

		ch = mueller_state.mueller_selected_channel
		ch_idx_map = {"B": 0, "G": 1, "R": 2}
		ch_idx = ch_idx_map.get(ch, 2)
		mats = frame[:, :, ch_idx, :, :]  # (H,W,4,4)
	else:
		return

	H, W, _, _ = mats.shape
	mats_flat = mats.reshape(-1, 4, 4)  # (H*W, 4, 4)

	if not mueller_hist_selected:
		selected_labels = mueller_elements[:]
	else:
		selected_labels = [lab for lab in mueller_elements if lab in mueller_hist_selected]

	if not selected_labels:
		return

	data_list = []
	labels = []
	for lab in selected_labels:
		i = int(lab[1])
		j = int(lab[2])
		data_list.append(mats_flat[:, i, j])
		labels.append(lab)

	# 3) Matplotlib figure 생성
	fig_width_inch = (CANVAS_WIDTH // 2) / 100.0  # DPI=100 기준
	fig_height_inch = CANVAS_HEIGHT / 100.0
	fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))

	ax.set_title(f"Mueller Matrix Element Distributions - {ch} Channel")
	ax.set_xlabel("Value")
	ax.set_ylabel("Number of Pixels")

	bins = 200
	value_range = (-1.0, 1.0)
	cmap = plt.get_cmap("tab20")

	for idx, data in enumerate(data_list):
		hist, bin_edges = np.histogram(data, bins=bins, range=value_range)
		centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
		ax.plot(centers, hist, label=labels[idx], color=cmap(idx % cmap.N))

	ax.legend(ncol=4, fontsize=8)
	plt.tight_layout()

	canvas = FigureCanvas(fig)
	canvas.draw()
	img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
	w, h = canvas.get_width_height()
	img = img.reshape(h, w, 4).astype(np.float32) / 255.0

	img_resized = cv2.resize(img, (CANVAS_WIDTH, CANVAS_HEIGHT))

	if dpg.does_item_exist(texture_tag):
		dpg.set_value(texture_tag, img_resized.flatten())

	common_state.last_figure = fig  # Export 버튼이 히스토그램도 저장 가능하도록 유지
	plt.close(fig)

def mueller_histogram():
	global mueller_hist_open
	mueller_hist_open = True
	dpg.configure_item("mueller_histogram_update", enabled=True, show=True)
	dpg.configure_item("mueller_histogram_close", enabled=True, show=True)
	enable_mueller_hist_side_by_side()
	render_mueller_histogram_to_texture("histogram_texture")

def close_mueller_histogram(sender, app_data, user_data):
	global mueller_hist_open
	mueller_hist_open = False
	if not dpg.does_item_exist("center_split_container"):
		return

	dpg.delete_item("center_split_container", children_only=True)

	with dpg.child_window(parent="center_split_container", tag="center_left_panel",
						  width=CANVAS_WIDTH, height=CANVAS_HEIGHT, no_scrollbar=True):
		dpg.add_image("uploaded_texture", width=CANVAS_WIDTH, height=CANVAS_HEIGHT)

	dpg.configure_item("mueller_histogram_update", enabled=False, show=False)
	dpg.configure_item("mueller_histogram_close", enabled=False, show=False)


try:
	mueller_video.on_after_redraw = update_mueller_hist_if_open
except Exception as e:
	print("[callbacks] failed to set mueller_video.on_after_redraw", e)