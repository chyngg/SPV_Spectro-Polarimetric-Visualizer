import scipy.io as spio
from scipy.interpolate import PchipInterpolator
from bisect import bisect
import numpy as np

def HSI2RGB(wY, stokes_data, d, threshold):
	# Load reference illuminant
	D = spio.loadmat('assets/illuminant_file/D_illuminants.mat')
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

	# Extract s0
	s0 = stokes_data

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