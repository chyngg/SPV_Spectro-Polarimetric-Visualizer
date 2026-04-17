import dearpygui.dearpygui as dpg
import numpy as np
import os
import sqlite3
import tkinter as tk
from tkinter import filedialog
from subs import common_state
from subs.Mueller_matrix_image import mueller_state, mueller_video
from subs.SP_image import sp_state
from subs.SP_image.sp_visualization import update_visualization, update_wavelengths_visualization
from subs.Mueller_matrix_image.mueller_visualization import visualize_rgb_mueller_grid, visualize_rgb_mueller_rgbgrid, visualize_decomposition

def open_import_dialog():
	root = tk.Tk()
	root.withdraw()

	file_path = filedialog.askopenfilename(
		title = "Select a file",
		filetypes = [("NumPy files", "*.npy *.npz")]
	)

	if not file_path:
		return

	if file_path.endswith('.npz'):
		open_npz_selection_dialog(file_path)
	else:
		add_file_to_history(file_path)
		common_state.selected_file_name = os.path.basename(file_path)
		load_npy_with_history(file_path)

def open_npz_selection_dialog(file_path):
	try:
		npz_file = np.load(file_path, mmap_mode='r')
		keys = npz_file.files
	except Exception as e:
		error_msg = f"File: {file_path}\nReason: {str(e)}"
		print(f"ERROR: {error_msg}")
		show_error_popup("Data Loading Error", error_msg)
		return

	if not keys:
		return

	if len(keys) == 1:
		_load_npz_key(file_path, keys[0])
		return

	if dpg.does_item_exist("npz_select_window"):
		dpg.delete_item("npz_select_window")

	with dpg.window(label="Select Array from NPZ", tag="npz_select_window", modal=True, show=True, width=350, height=150):
		dpg.add_text(f"File: {os.path.basename(file_path)}")
		dpg.add_combo(items=keys, default_value=keys[0], tag="npz_combo_selection", width=-1)
		dpg.add_spacer(height=10)

		def on_select():
			selected_key = dpg.get_value("npz_combo_selection")
			dpg.delete_item("npz_select_window")
			_load_npz_key(file_path, selected_key)

		with dpg.group(horizontal=True):
			dpg.add_button(label="Load", callback=on_select, width=160)
			dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("npz_select_window"), width=160)


def _load_npz_key(file_path, selected_key):
	history_path = f"{file_path}::{selected_key}"
	add_file_to_history(history_path)
	common_state.recent_files = [fp for fp in common_state.recent_files if fp != history_path]
	common_state.recent_files.insert(0, history_path)
	common_state.recent_files = get_recent_files(30)
	update_history_checkboxes()

	load_npy_and_display(file_path, selected_key)

def file_selected_callback(sender, app_data):
	if not app_data or 'file_path_name' not in app_data:
		dpg.delete_item("import_dialog_id")
		return

	selected_file_path = app_data['file_path_name']
	add_file_to_history(selected_file_path)
	common_state.selected_file_name = os.path.basename(selected_file_path)
	load_npy_with_history(selected_file_path)

	if dpg.does_item_exist("import_dialog_id"):
		dpg.delete_item("import_dialog_id")

def open_export_dialog():
	root = tk.Tk()
	root.withdraw()

	save_path = filedialog.asksaveasfilename(
		defaultextension = ".png",
		filetypes = [("PNG Image", "*.png")],
		title = "Save Image As"
	)

	if not save_path:
		return

	if common_state.last_figure:
		common_state.last_figure.savefig(save_path, dpi=300, bbox_inches='tight')

def export_image_callback(app_data):
	save_path = app_data
	if common_state.last_figure:
		common_state.last_figure.savefig(save_path, dpi=300, bbox_inches='tight')

def load_npy_and_display(file_path=None, npz_key=None):
	if not file_path:
		return

	try:
		mueller_video.player.detach_frames()
		if dpg.does_item_exist("video_controls"):
			dpg.configure_item("video_controls", show=False)
	except Exception as e:
		error_msg = f"File: {file_path}\nReason: {str(e)}"
		print(f"ERROR: {error_msg}")
		show_error_popup("Data Loading Error", error_msg)

	try:
		if npz_key:
			npz_data = np.load(file_path, mmap_mode='r')
			raw = npz_data[npz_key]
			common_state.selected_file_name = f"{os.path.basename(file_path)} [{npz_key}]"
		else:
			raw = np.load(file_path, mmap_mode='r')
			common_state.selected_file_name = os.path.basename(file_path)

		dim = raw.ndim

		if dim == 6 and raw.shape[-3:] == (3, 4, 4): # Mueller-matrix video
			common_state.current_tab = "Mueller_video"
			(common_state.vmin, common_state.vmax) = (-1, 1)
			dpg.configure_item("input_stokes_group", show=True)
			dpg.configure_item("stokes_custom_group", show=False)
			common_state.npy_data = raw
			update_tools_tab_from_current_tab()
			mueller_video.player.attach_frames(raw)

			dpg.configure_item("mueller_channel", enabled=True)
			dpg.configure_item("mueller_correction", enabled=True)
			dpg.configure_item("Mueller_rgb_positive", enabled=True)
			dpg.configure_item("Mueller_rgb_negative", enabled=True)
			dpg.configure_item("mueller_histogram", enabled=True)
			dpg.configure_item("center_status_fileinfo", show=False)
			dpg.configure_item("video_controls", show=True)

			if dpg.does_item_exist("channel_order_radio"):
				dpg.set_value("channel_order_radio", "RGB")

			update_wavelength_options()
			return

		# 비디오가 아니면 수정 가능하도록 메모리에 카피
		raw = np.array(raw)
		common_state.npy_data = raw

		if dim >= 3:
			missing_mask = np.isnan(raw) | (raw < -1e6) | (raw > 1e6)
			reduce_axes = tuple(range(2, raw.ndim))
			missing_pixels = np.any(missing_mask, axis=reduce_axes)  # (H, W)
			raw[missing_pixels, ...] = 0
			common_state.npy_data = raw

		if dim == 4 and raw.shape[2] == 4 and raw.shape[3] == 3: # Trichromatic
			common_state.current_tab = "Trichromatic"
			update_tools_tab_from_current_tab()
			dpg.configure_item("center_status_fileinfo", show=True)
			dpg.configure_item("input_stokes_group", show=False)
			dpg.configure_item("stokes_custom_group", show=False)
			if not sp_state.visualizing_by_wavelength:
				update_visualization(sp_state.sp_visualizing)
			else:
				update_wavelengths_visualization(sp_state.selected_wavelength, sp_state.selected_stokes)

		elif dim == 4 and raw.shape[2] == 4 and raw.shape[3] > 3: # Hyperspectral
			common_state.current_tab = "Hyperspectral"
			update_tools_tab_from_current_tab()
			dpg.configure_item("center_status_fileinfo", show=True)
			dpg.configure_item("input_stokes_group", show=False)
			dpg.configure_item("stokes_custom_group", show=False)
			update_visualization("original_hyper")

		elif dim == 5 and raw.shape[2:] == (3, 4, 4): # RGB Mueller image
			common_state.current_tab = "Mueller_image"
			update_tools_tab_from_current_tab()
			(common_state.vmin, common_state.vmax) = (-1, 1)
			dpg.configure_item("mueller_channel", enabled=True)
			dpg.configure_item("mueller_correction", enabled=True)
			dpg.configure_item("Mueller_rgb_positive", enabled=True)
			dpg.configure_item("Mueller_rgb_negative", enabled=True)
			dpg.configure_item("mueller_histogram", enabled=True)
			dpg.configure_item("center_status_fileinfo", show=True)
			dpg.configure_item("input_stokes_group", show=True)
			if mueller_state.visualization_mode == "Decomposition":
				visualize_decomposition(raw, channel=mueller_state.mueller_selected_channel, param_name=mueller_state.selected_decomposition)
			else:
				if mueller_state.mueller_visualizing in ["Original", "m00", "Gamma", "m00 (Keep Intensity)"]:
					visualize_rgb_mueller_grid(raw, channel=mueller_state.mueller_selected_channel,
											   correction=mueller_state.mueller_visualizing,
											   vmin=common_state.vmin, vmax=common_state.vmax)
				else:
					visualize_rgb_mueller_rgbgrid(raw, correction=mueller_state.mueller_selected_correction,
												  sign=mueller_state.mueller_visualizing)

		elif dim == 2:
			common_state.current_tab = "Trichromatic"
			update_tools_tab_from_current_tab()
			dpg.configure_item("center_status_fileinfo", show=True)
			update_visualization("s0")
		else:
			print("Unsupported data format: ", raw.shape)
			return

		dpg.set_value("status_file_name", common_state.selected_file_name)
		dpg.set_value("status_file_type", common_state.current_tab)
		if dpg.does_item_exist("channel_order_radio"):
			dpg.set_value("channel_order_radio", "RGB")
		update_wavelength_options()

	except Exception as e:
		print(f"UNEXPECTED ERROR: str({e})")
		show_error_popup("Unexpected Error", str(e))

def update_wavelength_options():
	if common_state.current_tab == "Hyperspectral":
		new_items = [f"{w}nm" for w in np.arange(450, 651, 10)]
	else:
		new_items = ["R", "G", "B"]

	common_state.selected_wavelengths = new_items
	common_state.selected_wavelength = new_items[0]
	dpg.configure_item("wavelength_options", items=new_items)
	dpg.set_value("wavelength_options", new_items[0])

def load_npy_with_history(file_path):
	if not file_path:
		return

	if "::" in file_path:
		path, key = file_path.split("::")
		_load_npz_key(path, key)
		return

	if file_path.endswith('.npz'):
		open_npz_selection_dialog(file_path)
		return

	common_state.recent_files = [fp for fp in common_state.recent_files if fp != file_path]
	common_state.recent_files.insert(0, file_path)
	common_state.recent_files = get_recent_files(30)
	load_npy_and_display(file_path)
	update_history_checkboxes()

def update_history_checkboxes():
	dpg.delete_item("checkbox_area", children_only=True)
	common_state.checked_files.clear()
	for specific_file_path in common_state.recent_files:
		filename = os.path.basename(specific_file_path)
		dpg.add_checkbox(
			label=filename,
			parent="checkbox_area",
			callback=make_checkbox_callback(specific_file_path)
		)

def load_file_callback():
	if len(common_state.checked_files) != 1:
		return
	file_path = common_state.checked_files[0]
	if "::" in file_path:
		path, key = file_path.split("::")
		load_npy_and_display(path, key)
	else:
		load_npy_and_display(file_path)

def make_checkbox_callback(file_path):
	return lambda s, a: checkbox_callback(s, a, file_path)

def checkbox_callback(sender, app_data, file_path):
	if app_data:
		if file_path not in common_state.checked_files:
			common_state.checked_files.append(file_path)
	else:
		if file_path in common_state.checked_files:
			common_state.checked_files.remove(file_path)


def add_file_to_history(file_path):
	conn = sqlite3.connect("history.db")
	c = conn.cursor()

	actual_path = file_path.split("::")[0]
	npz_key = file_path.split("::")[1] if "::" in file_path else None

	try:
		if npz_key:
			npz_file = np.load(actual_path, mmap_mode="r")
			arr = npz_file[npz_key]
		else:
			arr = np.load(actual_path, mmap_mode="r")

		data_type = "Trichromatic" if int(arr.shape[-1]) == 3 else "Hyperspectral"

		c.execute("""
					INSERT OR REPLACE INTO file_history (path, data_type, timestamp) 
					VALUES (?, ?, CURRENT_TIMESTAMP)
				""", (file_path, data_type))
		conn.commit()
	except Exception as e:
		print(f"Failed to add history: {e}")
		show_error_popup("Error", str(e))
	finally:
		conn.close()

def get_recent_files(limit=40):
	conn = sqlite3.connect("history.db")
	c = conn.cursor()
	c.execute("""
		SELECT path FROM file_history ORDER BY timestamp DESC LIMIT ?
	""", (limit,))
	files = [row[0] for row in c.fetchall()]
	conn.close()
	return files

def update_tools_tab_from_current_tab():
	try:
		if common_state.current_tab in ("Mueller_video", "Mueller_image"):
			dpg.set_value("tools_tab_bar", "tools_tab_mm")
		else:
			dpg.set_value("tools_tab_bar", "tools_tab_sp")
	except Exception as e:
		print(f"ERROR: str({e})")
		show_error_popup("Data Loading Error", str(e))

def show_error_popup(title, message):
	if dpg.does_item_exist("error_window"):
		dpg.delete_item("error_window")

	with dpg.window(label=title, modal=True, tag="error_window", no_title_bar=False, autosize=True):
		dpg.add_text(f"Error occurred: \n\n{message}", color=[255, 100, 100])
		dpg.add_separator()
		with dpg.group(horizontal=True):
			dpg.add_button(label="OK", width=75, callback=lambda: dpg.delete_item("error_window"))
			dpg.add_button(label="Copy to Clipboard", width=150,
						   callback=lambda: dpg.set_clipboard_text(message))