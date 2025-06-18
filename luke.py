# Uses command line to run the optimiser and does no data processing.
"""
Plan:
1. Ask what objective we target (pain, effort, or instability)
2. Ask for the height, weight, name, and age
3. Ask about # of trials then collect # of trials by asking the user for previous parameters
   and a score from 0 - 10 (0 is best) for how painful, how much effort, or how much instability.
   After each trial except for the last, we want to use BO to output the best
   next crutch parameters.
4. Generate graphics of the optimization process both 2D and 3D
5. We want to print the table of all trials and write it to csv file.
   luke_trials.csv. while also printing the graphs.
"""
# --------------SETUP ------------------------------------------------- #
from __future__ import annotations
import sys
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3-D proj)

# Tries to import GPy and GPyOpt and outputs an error if it fails.
try:
    import GPy
    from GPyOpt.methods import BayesianOptimization
except ImportError:
    print("GPy and GPyOpt are required. Install with:\n"
          "    python3 -m pip install GPy GPyOpt\n", file=sys.stderr)
    sys.exit(1)

# -------------- HELPER FUNCTIONS ------------------------------------------ #

"""This function just forces the user to enter a number and not some random string so it 
can convert the user input into a float """

def ask_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a number")


def round_float(value, bounds):

    """We want to round the user's input to the nearest allowed value, whether this
    is a normal rounded value (to the correct step size)
    or the bounds of alpha, beta, or gamma. Here I used
    key to tell the built in min function to compare the distance between any allowed
    value and the input 'value'"""

    return min(bounds, key=lambda x: abs(x - value))

# ------------------ BO SETUP --------------------------------------- #

""" This is the constraints on the input parameters. The third digit is the step size (it won't
suggest anything shorter than a 5 degree change in alpha/beta vs a 3 degree change in gamma)"""

alpha_range = list(range(70, 125, 5))   
beta_range  = list(range(90, 145, 5))   
gamma_range = list(range(0, 33, 3))    

"""Define the search space for GPyOpt similar to the way Riccardo did it.
GPyOpt needs a dictionary to tell it where to search. 
Dictionaries use keys and values to store data."""

SEARCH_SPACE = [
    {'name': 'alpha', 'type': 'discrete', 'domain': alpha_range},
    {'name': 'beta',  'type': 'discrete', 'domain': beta_range},
    {'name': 'gamma', 'type': 'discrete', 'domain': gamma_range}
]

class CrutchBO:

    """Bayesian-optimisation helper that stores trials in memory and suggests
    the next α, β, γ combination. In earlier versions of the project the optimiser
    was run on a full DataFrame; here we build it incrementally by appending to
    two lists (`_X`, `_Y`)."""
    
    def __init__(self):
        # self._X has lists of arrays of the crutch parameters
        self._X = []
        # self._Y has lists of arrays of the loss
        self._Y = []
        # Our predefined kernel
        self._kernel = GPy.kern.Matern52(input_dim=3, variance=1.0, lengthscale=1.0)

# To record a trial we just append the parameters and loss to the lists. 
    def record_trial(self, alpha, beta, gamma, loss):
        self._X.append([alpha, beta, gamma])
        self._Y.append([loss])

    def suggest_next(self) -> tuple[int, int, int]: # We suggest next best parameters as a tuple
        if not self._X:
            # If no trials yet, suggest a random point. B/c no trials means empty list.
            return (np.random.choice(alpha_range),
                    np.random.choice(beta_range),
                    np.random.choice(gamma_range))
# GPyOpt needs X and Y to be arrays. So we just convert the lists to arrays. 
        X = np.array(self._X)
        Y = np.array(self._Y)

        # Dummy objective always returns 0 (real data provided via X, Y)
        def objective(x):
            return np.array([[0]])

        bo = BayesianOptimization(
            f=objective,               # dummy objective
            domain=SEARCH_SPACE,
            model_type='GP',
            acquisition_type='EI',
            exact_feval=True,
            initial_design_numdata=0,
            X=X,
            Y=Y,
            kernel=self._kernel
        )

        bo.run_optimization(max_iter=1)
        a, b, g = bo.x_opt

        # Round to nearest allowed values
        a = round_float(a, alpha_range)
        b = round_float(b, beta_range)
        g = round_float(g, gamma_range)
        return a, b, g

"""Acutal code starts here:"""

print("\nLuke's BO Optimiser")

# Ask for objective we are minimizing
objectives = {"pain", "effort", "instability"}
while True:
    objective = input("What are we minimising? [pain / effort / instability]: ").strip().lower()

    """.strip and .lower ensures that user input is case insensitive
    we check if what they input is in the list of objectives"""

    if objective in objectives:
        break
    # otherwise we prompt them to re enter one of the above
    print("  → please type 'pain', 'effort' or 'instability'.")

# Ask for participant name and demographics
participant_name = input("\nParticipant name: ").strip()
print("\nEnter participant demographics (constants):")
height = ask_float("  Height (cm):  ")
weight = ask_float("  Weight (kg):  ")
age    = ask_float("  Age (years):  ")

# Asks how many trials to run
while True:
    n_trials = ask_float("\nHow many trials do you want to run?: ")
    # make sure user entered a positive integer
    if n_trials.is_integer() and n_trials > 0:
        n_trials = int(n_trials)
        break
    print("  → please enter a positive whole number.")

# DataFrame to store the trails in the experiment. The columns are the parameters and loss.
cols = [
    "alpha", "beta", "gamma", "height", "weight", "age", "loss",
]
# Create a data frame with 7 columns with those names
df = pd.DataFrame(columns=cols)

# instantiate bo.
bo = CrutchBO()

"""We have to run the loop for the number of trails the user wants to run. 
 Note that the range function defaults to start at 0, so we have to have 1, n_trials + 1."""

for trial in range(1, n_trials + 1):
    # use print(f' ) to print a string with good formatting
    print(f"\n=== Trial {trial} / {n_trials} ===")
    print("Enter last tested crutch parameters (degrees):")
    alpha = int(ask_float("  Alpha (70-120):  "))
    beta  = int(ask_float("  Beta  (90-140):  "))
    gamma = int(ask_float("  Gamma   (0-30):  "))
# asks the user the rating on a scale of 0-10 and enforces the bounds.
    while True:
        score = ask_float(f"\n{objective.capitalize()} score (0-10, lower = better): ")
        if 0 <= score <= 10:
            break
        print("  → please enter a value between 0 and 10.")

    """save the data in a row. Note AI told me to use a dictionary instead of a list because
    just in case we switch the order"""

    row = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "height": height,
        "weight": weight,
        "age": age,
        "loss": score,
    }
    
    """add the row to our data frame using the concat function. 
    basically we create a separate data frame with the row and connect them."""

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Remember this appends the parameters and the loss to the BO model. 
    bo.record_trial(alpha, beta, gamma, score)

    # if we are not on the last trial we continue running the BO 
    if trial < n_trials:
        # we have na, nb, and ng as the next parameters.
        na, nb, ng = bo.suggest_next()
        print("\n→ Suggested next crutch parameters:")
        print(f"   Alpha = {na}°")
        print(f"   Beta  = {nb}°")
        print(f"   Gamma = {ng}°")

# this is the end of the experiment when trial = n_trials
print("\n=== End of Experiment – Summary ===")
# prints the data frame in a table without the index numbers.
print(df[["alpha", "beta", "gamma", "loss"]].to_string(index=False))

# specifies a path to save the trials to. 
out_path = Path("luke_trials.csv")
df.to_csv(out_path, index=False)
# prints number of trials saved to that path.
print(f"\nSaved {len(df)} trials to {out_path.resolve()}")

"""All the plotting code"""

# Colour map normalisation (red = high loss, green = low)
cmap = plt.get_cmap("RdYlGn_r") # gets mat plot lib color map that translates numeric values to color
""" This normalizes the color map to the range of our loss values. 
Basically we get the lowest and highest loss values and normalize the color map to that range"""
norm = mpl.colors.Normalize(vmin=df["loss"].min(), vmax=df["loss"].max())

# -------- 2-D scatter plots --------
# creates the 1 row x 3 columns of sub plots. 
fig2d, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
# defines the axes plotted against each other with x axis first in the tuple
pairs = [("alpha", "beta"), ("gamma", "alpha"), ("gamma", "beta")]

for ax2d, (x_col, y_col) in zip(axes, pairs): # zip pairs up the axes and pairs
    # we get the associated pair of columns of the data frame that we want to plot
    ax2d.scatter(df[x_col], df[y_col], c=df["loss"], cmap=cmap, norm=norm, s=80, edgecolors="k")
    # this is getting the index of every row and the x and y values of that row in the data frame
    for idx, (xv, yv) in df[[x_col, y_col]].iterrows(): # we get the index and value of each row
        # this labels the points with the trial number note we add 1 because we have 0 indexed rows
        ax2d.text(xv, yv, str(idx + 1), fontsize=8, ha="center", va="center", color="white")
    ax2d.set_xlabel(x_col.capitalize())
    ax2d.set_ylabel(y_col.capitalize())
    ax2d.set_title(f"{y_col.capitalize()} vs {x_col.capitalize()}")

fig2d.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), label="Loss (higher = worse)")

# Create 3-D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# this plots the points in 3D space. the function scatter does it for us.
sc = ax.scatter(df["alpha"], df["beta"], df["gamma"],
                c=df["loss"], cmap=cmap, norm=norm, s=80, depthshade=True)

# Label each point with its trial number
for idx, (xa, xb, xg) in df[["alpha", "beta", "gamma"]].iterrows():
    ax.text(xa, xb, xg, str(idx + 1), fontsize=8, ha="center", va="center", color="black")

# Draw lines between successive trials to visualise the optimisation path
for i in range(len(df) - 1): # don't have a line to draw after the last point
    ax.plot([df["alpha"][i], df["alpha"][i + 1]],
            [df["beta"][i],  df["beta"][i + 1]],
            [df["gamma"][i], df["gamma"][i + 1]],
            color="gray", alpha=0.6)

# Axis labels
ax.set_xlabel("Alpha (°)")
ax.set_ylabel("Beta (°)")
ax.set_zlabel("Gamma (°)")

# Title with participant details
ax.set_title(f"{participant_name}: H {height:.0f} cm, W {weight:.0f} kg, Age {age:.0f}\nObjective: {objective.capitalize()}")

# Add colour bar
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, label="Loss (higher = worse)")

plt.show(block=False)  # show 2-D and 3-D figures without blocking execution

# -------- GP contour plot (α–β slice) --------

"""Build and train a fresh GP model on all trials in order to have a map that predicts loss
based on all the previous trials we have run."""
X_train = df[["alpha", "beta", "gamma"]].values
Y_train = df[["loss"]].values
gp_kernel = GPy.kern.Matern52(input_dim=3, variance=1.0, lengthscale=1.0)
gp_model = GPy.models.GPRegression(X_train, Y_train, gp_kernel)
gp_model.optimize(messages=False)

# Finds the median tested gamma value that we will slice around.
gamma_slice = float(np.median(df["gamma"]))

# Create grid over alpha/beta ranges which is like 70, 75, ... for alpha and 90, 95,... for beta
aa, bb = np.meshgrid(alpha_range, beta_range)
grid_points = np.column_stack([aa.ravel(), bb.ravel(), np.full(aa.size, gamma_slice)])

# Predict loss over grid
mean_pred, _ = gp_model.predict(grid_points)
zz = mean_pred.reshape(aa.shape)

# Plot contour
plt.figure(figsize=(6, 5))
cont = plt.contourf(aa, bb, zz, levels=20, cmap=cmap)
plt.colorbar(cont, label="Predicted loss (GP)")
plt.scatter(df["alpha"], df["beta"], c=df["loss"], cmap=cmap, norm=norm, edgecolors="k", s=80)
for idx, (xa, xb) in df[["alpha", "beta"]].iterrows():
    plt.text(xa, xb, str(idx + 1), fontsize=8, ha="center", va="center", color="white")

plt.title(f"BO Predicted Loss – γ slice {gamma_slice:.0f}°")
plt.xlabel("Alpha (°)")
plt.ylabel("Beta (°)")

plt.show()  # final blocking show for contour figure

# -------- GP contour plot (α–β slice) using γ MODE --------

# Determine mode; if multiple modes, take the first
gamma_mode = float(df["gamma"].mode().iloc[0])

# If mode equals median we already plotted it; skip duplication
if gamma_mode != gamma_slice:
    grid_points_mode = np.column_stack([
        aa.ravel(), bb.ravel(), np.full(aa.size, gamma_mode)
    ])
    mean_mode, _ = gp_model.predict(grid_points_mode)
    zz_mode = mean_mode.reshape(aa.shape)

    plt.figure(figsize=(6, 5))
    cont2 = plt.contourf(aa, bb, zz_mode, levels=20, cmap=cmap)
    plt.colorbar(cont2, label="Predicted loss (GP)")
    plt.scatter(df["alpha"], df["beta"], c=df["loss"], cmap=cmap, norm=norm,
                edgecolors="k", s=80)
    for idx, (xa, xb) in df[["alpha", "beta"]].iterrows():
        plt.text(xa, xb, str(idx + 1), fontsize=8, ha="center", va="center", color="white")

    plt.title(f"BO Predicted Loss – γ mode slice {gamma_mode:.0f}°")
    plt.xlabel("Alpha (°)")
    plt.ylabel("Beta (°)")

    plt.show()
