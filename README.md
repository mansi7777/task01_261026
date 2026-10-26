# Neural Tuning Curve Analysis

This assignment analyzes V1 neural responses to drifting gratings with eight orientations. The code converts spike counts into firing rates, estimates orientation tuning curves, and compares polynomial, Gaussian, and von Mises models.

## Requirements

- Python 3.10 or newer
- `h5py`
- `numpy`
- `matplotlib`
- `scipy`

Install the required packages with:

```bash
python -m pip install h5py numpy matplotlib scipy
```

## Dataset

The raw neural recording dataset is not included in this repository because it is large. Download the dataset from the Dryad repository:

<https://doi.org/10.5061/dryad.h9w0vt4s0>

The script expects the HDF5 MATLAB file
`107l003p143_spikes.mat` from the `dataset1` folder. After downloading it, update `file_path` near the top of `Task Code.py` so that it points to the location of the file on your computer. For example:

```python
file_path = Path(r"C:\path\to\dataset\dataset1\107l003p143_spikes.mat")
```

Do not add the downloaded dataset to version control.

## Running the analysis

Open a terminal in this assignment folder and run:

```bash
python "Task Code.py"
```

The script prints model parameters and mean-squared errors for each analysis part. It also opens the figures interactively and saves all generated graphs in the `figures` folder:

- `part1_tuning_curves.png`: measured mean firing-rate tuning curves for selected V1 neurons.
- `part2_polynomial_fits.png`: polynomial fits of degrees 1, 3, 4, and 7 for the selected neuron.
- `part3_gaussian_fit.png`: Gaussian tuning-curve fit.
- `part4_von_mises_fit.png`: circular von Mises tuning-curve fit.
- `part5_noisy_neurons.png`: linear, Gaussian, and von Mises comparisons for the two noisiest neurons.

The Gaussian and von Mises curves are evaluated on a dense orientation grid for plotting. The original eight orientation measurements are used for model fitting and MSE calculations.

## Relationship to the report

`Task Code.py` implements the analysis described in the assignment report:

1. **Raw tuning curves:** spike counts are summed over the 1280 ms stimulus period, converted to spikes per second, and averaged separately for each orientation.
2. **Polynomial regression:** linear, cubic, fourth-order, and seventh-order polynomial models are fitted to the selected neuron's mean responses.
3. **Gaussian regression:** a baseline, amplitude, preferred orientation, and width are estimated with nonlinear least squares.
4. **Von Mises regression:** the orientations are converted to radians and a circular tuning model is fitted using a concentration parameter.
5. **Model comparison and noise:** trial-to-trial variability is estimated within each orientation, the two noisiest neurons are selected, and their linear, Gaussian, and von Mises fits are compared.

The printed parameter values and MSEs should be used with the corresponding figures when writing or checking the report. The generated images in `figures` provide the visual results for each numbered analysis part.
