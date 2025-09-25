import dearpygui.dearpygui as dpg
import numpy as np
import os
import sqlite3
import tkinter as tk
from tkinter import filedialog
from subs import state
from .visualization import update_visualization
from .graph import show_combined_graph
from .callbacks import multi_graph_option_callback

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
	state.selected_file_name = os.path.basename(file_path)
	load_npy_with_history(file_path)

def file_selected_callback(sender, app_data):
	if not app_data or 'file_path_name' not in app_data:
		dpg.delete_item("import_dialog_id")
		return

	selected_file_path = app_data['file_path_name']
	add_file_to_history(selected_file_path)
	state.selected_file_name = os.path.basename(selected_file_path)
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

	if state.last_figure:
		state.last_figure.savefig(save_path, dpi=300, bbox_inches='tight')

def export_image_callback(app_data):
	save_path = app_data
	if state.last_figure:
		state.last_figure.savefig(save_path, dpi=300, bbox_inches='tight')

def load_npy_and_display(file_path=None):
	if not file_path:
		return

	try:
		state.npy_data = np.load(file_path)
		arr = state.npy_data
		missing_mask = np.isnan(arr) | (arr < -1e6) | (arr > 1e6)

		reduce_axes = tuple(range(2, arr.ndim))  # 5D면 (2,3,4), 4D면 (2,3)
		missing_pixels = np.any(missing_mask, axis=reduce_axes)  # (H, W)

		arr[missing_pixels, ...] = 0
		state.npy_data = arr
		dim = state.npy_data.ndim

		dpg.configure_item("mueller_channel", enabled=False)
		if dim == 4 and arr.shape[2] == 4 and arr.shape[3] == 3: # RGB
			state.current_tab = "Trichromatic"
			update_visualization("original")
		elif dim == 4 and arr.shape[2] == 4 and arr.shape[3] > 3: # Hyperpsectral
			state.current_tab = "Hyperspectral"
			update_visualization("original_hyper")
		elif dim == 5 and arr.shape[2:] == (3, 4, 4): # RGB Mueller
			state.current_tab = "RGB_Mueller"
			(state.vmin, state.vmax) = (-1, 1)
			from .visualization import visualize_rgb_mueller_grid
			dpg.configure_item("mueller_channel", enabled=True)
			dpg.configure_item("mueller_correction", enabled=True)
			dpg.configure_item("Mueller_rgb_positive", enabled=True)
			dpg.configure_item("Mueller_rgb_negative", enabled=True)
			visualize_rgb_mueller_grid(arr, channel=2, vmin=-1, vmax=1)
		else:
			print("Unsupported data format: ", arr.shape)
			return

		update_wavelength_options()

	except Exception as e:
		print(f"Unexpected error: {e}")

def update_wavelength_options():
	if state.current_tab == "Trichromatic":
		new_items = ["R", "G", "B"]
	else:
		new_items = [f"{w}nm" for w in np.arange(450, 651, 10)]

	state.selected_wavelengths = new_items
	state.selected_wavelength = new_items[0]
	dpg.configure_item("wavelength_options", items=new_items)
	dpg.set_value("wavelength_options", new_items[0])

def load_npy_with_history(file_path):
	if not file_path:
		return

	state.recent_files = [fp for fp in state.recent_files if fp != file_path]
	state.recent_files.insert(0, file_path)
	state.recent_files = get_recent_files(30)
	load_npy_and_display(file_path)
	update_history_checkboxes()

def update_history_checkboxes():
	dpg.delete_item("checkbox_area", children_only=True)

	for specific_file_path in state.recent_files:
		filename = os.path.basename(specific_file_path)
		dpg.add_checkbox(
			label=filename,
			parent="checkbox_area",
			callback=make_checkbox_callback(specific_file_path)
		)

def load_file_callback():
	if len(state.checked_files) != 1:
		return
	load_npy_and_display(state.checked_files[0])

def make_checkbox_callback(file_path):
	return lambda s, a: checkbox_callback(s, a, file_path)

def checkbox_callback(sender, app_data, file_path):
	if app_data:
		if file_path not in state.checked_files:
			state.checked_files.append(file_path)
	else:
		if file_path in state.checked_files:
			state.checked_files.remove(file_path)

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