from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file_path = Path(r"C:\Users\mansi\Downloads\dataset\dataset1\107l003p143_spikes.mat")
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

with h5py.File(file_path, "r") as f:
    spikes = [
        np.transpose(np.asarray(f[c]), (2, 1, 0))
        for c in np.asarray(f["spikes"]).flat
    ]
    stim = np.asarray(f["stim"]).reshape(-1)

# Use V1 recordings
v1_spikes = spikes[0]

# Convert spike counts during the 1280 ms stimulus presentation
# into firing rates in spikes per second.
# Shape changes from (neurons, time, trials) to (neurons, trials).
stimulus_duration_s = 1280 / 1000
trial_responses = v1_spikes.sum(axis=1) / stimulus_duration_s

print("\nPart 1: Raw Spike to Tuning Curves")
orientation_labels = np.sort(np.unique(stim))
orientations = (
    orientation_labels - orientation_labels.min()
) * (180 / len(orientation_labels))

# Average trial responses separately for each orientation.
# Result shape: (neurons, orientations).
mean_responses = np.array([
    trial_responses[:, stim == orientation].mean(axis=1)
    for orientation in orientation_labels
]).T

# Select neurons with the strongest orientation modulation for the figure.
# This is a reproducible screening rule, not a replacement for visual    inspection in the report.
modulation_score = (
    (mean_responses.max(axis=1) - mean_responses.min(axis=1))/
    (mean_responses.max(axis=1) + mean_responses.min(axis=1))
)
# Evry neuron's modulation score is between 0.0 and 1.0, so we can just select the first six neurons.
plot_neurons = np.argsort(modulation_score)[-6:][::-1]
selected_neuron = plot_neurons[0]

print(f"Session: {file_path.name}")
print(f"Number of V1 neurons: {v1_spikes.shape[0]}")
print(f"Number of trials per orientation: "
      f"{[int(np.sum(stim == label)) for label in orientation_labels]}")
print(f"Stimulus labels stored in the file: {orientation_labels}")
print(f"Stimulus orientations (degrees): {orientations}")
print(f"Mean response matrix shape: {mean_responses.shape}")
print(f"Selected neuron: {selected_neuron}")
print(f"Selected neuron modulation score: "
      f"{modulation_score[selected_neuron]:.3f}")

fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
for axis, neuron in zip(axes.flat, plot_neurons):
    axis.plot(orientations, mean_responses[neuron], marker="o")
    axis.set_title(f"Neuron {neuron}")
    axis.set_xlabel("Orientation (degrees)")
    axis.set_ylabel("Mean firing rate (spikes/s)")
    axis.grid(alpha=0.3)

fig.suptitle("V1 mean orientation tuning curves")
fig.tight_layout()
fig.savefig(figures_dir / "part1_tuning_curves.png", dpi=150)
plt.show()



### Part 2: polynomial regression for the selected neuron.
selected_responses = mean_responses[selected_neuron]
polynomial_degrees = (1, 3, 4, 7)
polynomial_models = {}
polynomial_mse = {}

for degree in polynomial_degrees:
    coefficients = np.polyfit(orientations, selected_responses, degree)
    predictions = np.polyval(coefficients, orientations)
    polynomial_models[degree] = coefficients
    polynomial_mse[degree] = np.mean((selected_responses - predictions) ** 2)

print("\nPart 2: Polynomial Regression")
print(f"Selected neuron responses: {selected_responses}")
for degree in polynomial_degrees:
    print(
        f"Degree {degree}: MSE = {polynomial_mse[degree]:.8f}; "
        f"parameters = {degree + 1}"
    )

orientation_grid = np.linspace(orientations.min(), orientations.max(), 400)
fig, axis = plt.subplots(figsize=(10, 6))
axis.scatter(
    orientations,
    selected_responses,
    color="black",
    zorder=3,
    label=f"Neuron {selected_neuron} data",
)

for degree in polynomial_degrees:
    axis.plot(
        orientation_grid,
        np.polyval(polynomial_models[degree], orientation_grid),
        label=f"Degree {degree} (MSE={polynomial_mse[degree]:.3g})",
    )

axis.set_title(f"Polynomial fits for V1 neuron {selected_neuron}")
axis.set_xlabel("Orientation (degrees)")
axis.set_ylabel("Mean firing rate (spikes/s)")
axis.grid(alpha=0.3)
axis.legend()
fig.tight_layout()
fig.savefig(figures_dir / "part2_polynomial_fits.png", dpi=150)
plt.show()


### Part 3: nonlinear Gaussian regression.
def gaussian_tuning(
    orientation: np.ndarray,
    baseline: float,
    amplitude: float,
    preferred_orientation: float,
    width: float,
) -> np.ndarray:
    return baseline + amplitude * np.exp(
        -((orientation - preferred_orientation) ** 2) / (2 * width**2)
    )


initial_parameters = (
    float(selected_responses.min()),
    float(selected_responses.max() - selected_responses.min()),
    float(orientations[np.argmax(selected_responses)]),
    1.0,
)
parameter_bounds = (
    (0.0, 0.0, orientations.min(), 1e-6),
    (
        np.inf,
        np.inf,
        orientations.max(),
        orientations.max() - orientations.min(),
    ),
)

fitted_parameters, _ = curve_fit(
    gaussian_tuning,
    orientations,
    selected_responses,
    p0=initial_parameters,
    bounds=parameter_bounds,
    maxfev=100_000,
)
gaussian_predictions = gaussian_tuning(orientations, *fitted_parameters)
gaussian_mse = np.mean((selected_responses - gaussian_predictions) ** 2)

baseline, amplitude, preferred_orientation, width = fitted_parameters
print("\nPart 3: Gaussian nonlinear regression")
print(f"Baseline (B): {baseline:.8f}")
print(f"Amplitude (A): {amplitude:.8f}")
print(f"Preferred orientation (mu): {preferred_orientation:.8f}")
print(f"Width (sigma): {width:.8f}")
print(f"Gaussian MSE: {gaussian_mse:.8f}")

fig, axis = plt.subplots(figsize=(10, 6))
axis.scatter(
    orientations,
    selected_responses,
    color="black",
    zorder=3,
    label=f"Neuron {selected_neuron} data",
)
axis.plot(
    orientation_grid,
    gaussian_tuning(orientation_grid, *fitted_parameters),
    color="tab:orange",
    label=f"Gaussian (MSE={gaussian_mse:.3g})",
)
axis.set_title(f"Gaussian tuning fit for V1 neuron {selected_neuron}")
axis.set_xlabel("Orientation (degrees)")
axis.set_ylabel("Mean firing rate (spikes/s)")
axis.grid(alpha=0.3)
axis.legend()
fig.tight_layout()
fig.savefig(figures_dir / "part3_gaussian_fit.png", dpi=150)
plt.show()


# Part 4: circular orientation and von Mises regression.
# The eight dataset labels are equally spaced over 0 to 180 degrees.
# Orientation is represented in radians for the trigonometric model.
orientation_degrees = orientations
orientation_radians = np.deg2rad(orientation_degrees)

def von_mises_tuning(
    orientation: np.ndarray,
    baseline: float,
    amplitude: float,
    preferred_orientation: float,
    concentration: float,
) -> np.ndarray:
    return baseline + amplitude * np.exp(
        concentration
        * np.cos(2 * (orientation - preferred_orientation))
    )


von_mises_initial_parameters = (
    float(selected_responses.min()),
    float((selected_responses.max() - selected_responses.min()) / np.e),
    float(orientation_radians[np.argmax(selected_responses)]),
    1.0,
)
von_mises_parameter_bounds = (
    (0.0, 0.0, 0.0, 0.0),
    (np.inf, np.inf, np.pi, 100.0),
)

von_mises_parameters, _ = curve_fit(
    von_mises_tuning,
    orientation_radians,
    selected_responses,
    p0=von_mises_initial_parameters,
    bounds=von_mises_parameter_bounds,
    maxfev=100_000,
)
von_mises_predictions = von_mises_tuning(
    orientation_radians,
    *von_mises_parameters,
)
von_mises_mse = np.mean(
    (selected_responses - von_mises_predictions) ** 2
)

vm_baseline, vm_amplitude, vm_preferred, vm_concentration = von_mises_parameters
print("\nPart 4: von Mises circular regression")
print(f"Orientation values (degrees): {orientation_degrees}")
print(f"Baseline (B): {vm_baseline:.8f}")
print(f"Amplitude (A): {vm_amplitude:.8f}")
print(f"Preferred orientation (mu): {np.rad2deg(vm_preferred):.8f} degrees")
print(f"Concentration (kappa): {vm_concentration:.8f}")
print(f"von Mises MSE: {von_mises_mse:.8f}")

orientation_grid_degrees = np.linspace(0.0, 180.0, 400)
orientation_grid_radians = np.deg2rad(orientation_grid_degrees)
fig, axis = plt.subplots(figsize=(10, 6))
axis.scatter(
    orientation_degrees,
    selected_responses,
    color="black",
    zorder=3,
    label=f"Neuron {selected_neuron} data",
)
axis.plot(
    orientation_grid_degrees,
    von_mises_tuning(orientation_grid_radians, *von_mises_parameters),
    color="tab:green",
    label=f"von Mises (MSE={von_mises_mse:.3g})",
)
axis.set_title(f"Von Mises tuning fit for V1 neuron {selected_neuron}")
axis.set_xlabel("Orientation (degrees)")
axis.set_ylabel("Mean firing rate (spikes/s)")
axis.set_xlim(0, 180)
axis.grid(alpha=0.3)
axis.legend()
fig.tight_layout()
fig.savefig(figures_dir / "part4_von_mises_fit.png", dpi=150)
plt.show()


# Part 5: compare models and test noisier neurons.
print("\nPart 5: model comparison for selected neuron")
model_summary = (
    ("Linear", 2, polynomial_mse[1], "No", "No"),
    ("Third-order polynomial", 4, polynomial_mse[3], "No", "No"),
    ("Fourth-order polynomial", 5, polynomial_mse[4], "No", "No"),
    ("Seventh-order polynomial", 8, polynomial_mse[7], "No", "No"),
    ("Gaussian", 4, gaussian_mse, "Yes", "No"),
    ("von Mises", 4, von_mises_mse, "Yes", "Yes"),
)
print("Model                         Parameters       MSE  Biological  Circular")
for name, parameters, mse, biological, circular in model_summary:
    print(
        f"{name:<29} {parameters:>10}  {mse:>10.6f}"
        f"  {biological:>10}  {circular:>8}"
    )

# Estimate noise from trial-to-trial variation around each neuron's
# orientation-specific mean response.
orientation_indices = np.array([
    np.flatnonzero(orientation_labels == stimulus)[0] for stimulus in stim
])
expected_trial_responses = mean_responses[:, orientation_indices]
within_orientation_noise = np.std(
    trial_responses - expected_trial_responses,
    axis=1,
)
noisy_neurons = [
    int(neuron)
    for neuron in np.argsort(within_orientation_noise)[::-1]
    if neuron != selected_neuron
][:2]

print("\nNoisy-neuron comparison")
print("Noise is the pooled within-orientation trial-response standard deviation.")
for neuron in [selected_neuron, *noisy_neurons]:
    print(
        f"Neuron {neuron}: noise SD = {within_orientation_noise[neuron]:.6f}"
    )


def fit_gaussian_responses(responses: np.ndarray) -> tuple[np.ndarray, float]:
    initial = (
        float(responses.min()),
        float(responses.max() - responses.min()),
        float(orientations[np.argmax(responses)]),
        1.0,
    )
    parameters, _ = curve_fit(
        gaussian_tuning,
        orientations,
        responses,
        p0=initial,
        bounds=parameter_bounds,
        maxfev=100_000,
    )
    predictions = gaussian_tuning(orientations, *parameters)
    return parameters, float(np.mean((responses - predictions) ** 2))


def fit_von_mises_responses(
    responses: np.ndarray,
) -> tuple[np.ndarray, float]:
    initial = (
        float(responses.min()),
        float((responses.max() - responses.min()) / np.e),
        float(orientation_radians[np.argmax(responses)]),
        1.0,
    )
    parameters, _ = curve_fit(
        von_mises_tuning,
        orientation_radians,
        responses,
        p0=initial,
        bounds=von_mises_parameter_bounds,
        maxfev=100_000,
    )
    predictions = von_mises_tuning(orientation_radians, *parameters)
    return parameters, float(np.mean((responses - predictions) ** 2))


noisy_model_results = {}
noisy_fit_parameters = {}
for neuron in noisy_neurons:
    responses = mean_responses[neuron]
    linear_coefficients = np.polyfit(orientations, responses, 1)
    linear_predictions = np.polyval(linear_coefficients, orientations)
    gaussian_parameters, gaussian_neuron_mse = fit_gaussian_responses(responses)
    von_mises_neuron_parameters, von_mises_neuron_mse = (
        fit_von_mises_responses(responses)
    )
    noisy_model_results[neuron] = {
        "Linear": (
            linear_predictions,
            float(np.mean((responses - linear_predictions) ** 2)),
        ),
        "Gaussian": (
            gaussian_tuning(orientations, *gaussian_parameters),
            gaussian_neuron_mse,
        ),
        "von Mises": (
            von_mises_tuning(
                orientation_radians,
                *von_mises_neuron_parameters,
            ),
            von_mises_neuron_mse,
        ),
    }
    noisy_fit_parameters[neuron] = {
        "Linear": linear_coefficients,
        "Gaussian": gaussian_parameters,
        "von Mises": von_mises_neuron_parameters,
    }
    print(f"\nNeuron {neuron}")
    for model_name, (_, mse) in noisy_model_results[neuron].items():
        print(f"{model_name:<10} MSE = {mse:.8f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for axis, neuron in zip(axes, noisy_neurons):
    responses = mean_responses[neuron]
    axis.scatter(orientations, responses, color="black", zorder=3)
    results = noisy_model_results[neuron]
    parameters = noisy_fit_parameters[neuron]
    axis.plot(
        orientation_grid,
        np.polyval(parameters["Linear"], orientation_grid),
        label=f"Linear (MSE={results['Linear'][1]:.3g})",
    )
    axis.plot(
        orientation_grid,
        gaussian_tuning(orientation_grid, *parameters["Gaussian"]),
        label=f"Gaussian (MSE={results['Gaussian'][1]:.3g})",
    )
    axis.plot(
        orientation_grid,
        von_mises_tuning(
            np.deg2rad(orientation_grid),
            *parameters["von Mises"],
        ),
        label=f"von Mises (MSE={results['von Mises'][1]:.3g})",
    )
    axis.set_title(
        f"Neuron {neuron} (noise SD={within_orientation_noise[neuron]:.2f})"
    )
    axis.set_xlabel("Orientation (degrees)")
    axis.grid(alpha=0.3)
    axis.legend()

axes[0].set_ylabel("Mean firing rate (spikes/s)")
fig.suptitle("Model comparison for noisier neurons")
fig.tight_layout()
fig.savefig(figures_dir / "part5_noisy_neurons.png", dpi=150)
plt.show()
