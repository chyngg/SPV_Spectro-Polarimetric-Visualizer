import dearpygui.dearpygui as dpg
import numpy as np
import os
from . import state
from .visualization import update_visualization

def open_import_dialog():
	if dpg.does_item_exist("import_dialog_id"):
		dpg.delete_item("import_dialog_id")

	with dpg.file_dialog(directory_selector=False, show=True, callback=file_selected_callback,
						 tag="import_dialog_id", width=800, height=400):
		dpg.add_file_extension(".npy", color=(0, 200, 255, 255))

def file_selected_callback(sender, app_data):
	if not app_data or 'file_path_name' not in app_data:
		dpg.delete_item("import_dialog_id")
		return

	selected_file_path = app_data['file_path_name']
	state.selected_file_name = os.path.basename(selected_file_path)
	load_npy_and_display(selected_file_path)

	if dpg.does_item_exist("import_dialog_id"):
		dpg.delete_item("import_dialog_id")

def open_export_dialog():
	with dpg.file_dialog(directory_selector=False, show=True, callback=export_image_callback, tag="export_dialog_id", width=800, height=400):
		dpg.add_file_extension(".png", color=(0, 200, 0, 255))

def export_image_callback(app_data):
	save_path = app_data
	if state.last_figure:
		state.last_figure.savefig(save_path, dpi=300, bbox_inches='tight')

def load_npy_and_display(file_path=None):
	if not file_path:
		return

	try:
		state.npy_data = np.load(file_path)
		missing_mask = np.isnan(state.npy_data) | (state.npy_data < -1e6) | (state.npy_data > 1e6)
		missing_pixels = np.any(missing_mask, axis=(2, 3))
		state.npy_data[missing_pixels, :, :] = 0
		last_dim = state.npy_data.shape[-1]

		if last_dim == 3:
			state.current_tab = "Trichromatic"
			update_visualization("original")
		else:
			state.current_tab = "Hyperspectral"
			update_visualization("original_hyper")
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