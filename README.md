# Visualization Interface for Spectral-Polarimetric datasets

---

[[Project Page]](https://eschoi.com/SPDataset/) [[Data]](https://huggingface.co/datasets/jyj7913/spectro-polarimetric)

![RGB_1.png](/example/RGB_1.png)

![Hyperspectral_1.png](/example/Hyperspectral_1.png)

## Requirements

---

To run this GUI application for Spectral-Polarimetric data visualization, the following Python packages must be installed.

```bash
pip install dearpygui dearpygui-extend matplotlib numpy opencv-python scipy
```

### Additional Requirements

This application loads Arial font from the Windows default path:

```jsx
C:/Windows/Fonts/arial.ttf
```

If you are on **macOS or Linux**, update the font path in `setup_fonts()` in `subs.py` to a valid path on your system:

- macOS: `/System/Library/Fonts/SFNS.ttf`
- Ubuntu: `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`

## How to Run


```bash
python main.py
```

## Features

---

- Visualize for entire wavelengths:
    - Full Stokes parameters (`s0`, `s1`, `s2` , `s3`)
    - Polarization feature maps (`DoLP`, `AoLP`, `DoCP`, `CoP`)
    - Unpolarized/Polarized light
    - Histogram visualization
        - Stokes vector distribution
        - Gradient distributions
        - Gradient derivatives distribution
        - Polarization feature gradients (`DoLP`, `AoLP`, `DoCP`, `CoP`)
- Visualize for individual wavelengths:
    - Full Stokes parameters (`s0`, `s1`, `s2` , `s3`)
    - Polarization feature maps (`DoLP`, `AoLP`, `DoCP`, `CoP`)
- RGB approximation from hyperspectral data
- Region-based graph plotting accross wavelengths
    - Stokes parameters (`s0`, `s1`, `s2` , `s3`)
    - Polarization features (`DoLP`, `AoLP`, `DoCP`, `CoP`)
- Save the visualization as `.png`