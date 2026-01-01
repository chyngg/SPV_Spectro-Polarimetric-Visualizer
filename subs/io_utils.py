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
from subs.Mueller_matrix_image.mueller_visualization import visualize_rgb_mueller_grid, visualize_rgb_mueller_rgbgrid

def open_import_dialog():
	root = tk.Tk()
	root.withdraw()

	file_path = filedialog.askopenfilename(
		title = "Select a .npy file",
		filetypes = [("NumPy Array files", "*.npy")]
	)

	if not file_path:
		return

	add_file_to_history(file_path)
	common_state.selected_file_name = os.path.basename(file_path)
	load_npy_with_history(file_path)

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

def load_npy_and_display(file_path=None):
	if not file_path:
		return

	try:
		mueller_video.player.detach_frames()
	except Exception:
		pass

	try:
		raw = np.load(file_path)
		common_state.npy_data = raw
		dim = raw.ndim
		common_state.selected_file_name = os.path.basename(file_path)

		if dim == 6 and raw.shape[-3:] == (3, 4, 4): # Mueller-matrix video
			common_state.current_tab = "Mueller_video"
			common_state.npy_data = raw
			update_tools_tab_from_current_tab()
			mueller_video.player.attach_frames(raw)

			dpg.configure_item("mueller_channel", enabled=True)
			dpg.configure_item("mueller_correction", enabled=True)
			dpg.configure_item("Mueller_rgb_positive", enabled=True)
			dpg.configure_item("Mueller_rgb_negative", enabled=True)
			dpg.configure_item("mueller_histogram", enabled=True)
			dpg.configure_item("center_status_fileinfo", show=False)

			update_wavelength_options()
			return


		if dim >= 3:
			missing_mask = np.isnan(raw) | (raw < -1e6) | (raw > 1e6)
			reduce_axes = tuple(range(2, raw.ndim))
			missing_pixels = np.any(missing_mask, axis=reduce_axes)  # (H, W)
			raw[missing_pixels, ...] = 0
			common_state.npy_data = raw

		if dim == 4 and raw.shape[2] == 4 and raw.shape[3] == 3: # SP_image - RGB
			common_state.current_tab = "Trichromatic"
			update_tools_tab_from_current_tab()
			dpg.configure_item("center_status_fileinfo", show=True)
			if not sp_state.visualizing_by_wavelength:
				update_visualization(sp_state.sp_visualizing)
			else:
				update_wavelengths_visualization(sp_state.selected_wavelength, sp_state.selected_stokes)

		elif dim == 4 and raw.shape[2] == 4 and raw.shape[3] > 3: # Hyperspectral
			common_state.current_tab = "Hyperspectral"
			update_tools_tab_from_current_tab()
			dpg.configure_item("center_status_fileinfo", show=True)
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
			if mueller_state.mueller_visualizing in ["Original", "m00", "Gamma"]:
				visualize_rgb_mueller_grid(raw, channel="R", correction=mueller_state.mueller_visualizing, vmin=-1, vmax=1)
			else:
				visualize_rgb_mueller_rgbgrid(raw, correction=mueller_state.mueller_selected_correction, sign=mueller_state.mueller_visualizing)

		elif dim == 2:
			common_state.current_tab = "Trichromatic"
			update_tools_tab_from_current_tab()
			dpg.configure_item("center_status_fileinfo", show=True)
			update_visualization("s0")
		else:
			print("Unsupported data format: ", raw.shape)
			return

		print(common_state.current_tab)
		dpg.set_value("status_file_name", common_state.selected_file_name)
		dpg.set_value("status_file_type", common_state.current_tab)
		update_wavelength_options()

	except Exception as e:
		print(f"Unexpected error: {e}")

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

	common_state.recent_files = [fp for fp in common_state.recent_files if fp != file_path]
	common_state.recent_files.insert(0, file_path)
	common_state.recent_files = get_recent_files(30)
	load_npy_and_display(file_path)
	update_history_checkboxes()

def update_history_checkboxes():
	dpg.delete_item("checkbox_area", children_only=True)

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
	load_npy_and_display(common_state.checked_files[0])

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
	arr = np.load(file_path, mmap_mode="r")
	if int(arr.shape[-1]) == 3:
		data_type = "Trichromatic"
	else:
		data_type = "Hyperspectral"

	c.execute("""
				INSERT OR REPLACE INTO file_history (path, data_type, timestamp) 
				VALUES (?, ?, CURRENT_TIMESTAMP)
			""", (file_path, data_type))
	conn.commit()
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
	except Exception:
		pass