from SP_image import state
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import dearpygui.dearpygui as dpg

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

	#  Matplotlib figure → DearPyGui
	canvas = FigureCanvas(fig)
	canvas.draw()

	image_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(canvas.get_width_height()[::-1] + (4,))
	image_array = image_array.astype(np.float32) / 255.0  # Normalize (0~1)

	dpg.set_value("uploaded_texture", image_array.flatten())
	plt.close(fig)

def show_stokes_histogram(parameter, direction="X"):
	state.shown_histogram = parameter
	if state.npy_data is None or state.current_tab == "RGB_Mueller":
		return

	s0, s1, s2, s3 = state.npy_data[:, :, :, 0].transpose(2, 0, 1)

	dpg.configure_item("polarimetric_options", enabled=False)
	dpg.configure_item("wavelength_options", enabled=False)

	if parameter == 1: # Distribution of Stokes vector
		stokes_labels = ["$s_0$", "$s_1$", "$s_2$", "$s_3$"]
		colors = ['blue', 'orange', 'green', 'red']
		data_flattened = state.npy_data.reshape(-1, state.npy_data.shape[2])
		data_list = [data_flattened[:, i] for i in range(data_flattened.shape[1])]
		generate_histogram(data_list, stokes_labels, colors, title="Stokes Distribution", xlabel="Value", ylabel="Number of Pixels")


	if parameter == 2: # Gradient distribution of Stokes vector
		stokes_labels = ["$s_0$", "$s_1$", "$s_2$", "$s_3$"]
		colors = ['blue', 'orange', 'green', 'red']
		gradients_x = []
		gradients_y = []
		for i in range(4):  # s0, s1, s2, s3
			grad_x, grad_y = np.gradient(state.npy_data[:, :, i, 0])  # Gradient of the first channel
			gradients_x.append(grad_x.flatten())  # Flatten the X-direction gradient
			gradients_y.append(grad_y.flatten())  # Flatten the Y-direction gradient

		if direction == "X":
			generate_histogram(gradients_x, stokes_labels, colors, title="Gradient Distributions of Stokes Vector(X direction)", xlabel="Gradient (X direction)", ylabel="Log Probability", value_range=(-3, 3))

		else: # direction Y
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

		else: # direction Y
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
			else: # direction Y
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
