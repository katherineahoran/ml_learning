import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Ellipse
from scipy import linalg
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# Generate Data

n_samples = # TODO
np.random.seed(0)
C = # TODO
component_1 = # TODO
component_2 = # TODO

X = # TODO

# Fit Data

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


param_grid = # TODO
    "n_components": range(1, 7),
    "covariance_type": ["spherical", "tied", "diag", "full"],
}
grid_search = # TODO
    GaussianMixture(), param_grid= # TODO
)
grid_search.fit(X)

# Plot best estimator with data

color_iter = # TODO
Y_ = # TODO

fig, ax = # TODO

for i, (mean, cov, color) in enumerate(
    zip(
        grid_search.best_estimator_.means_,
        grid_search.best_estimator_.covariances_,
        color_iter,
    )
):
    v, w = # TODO
    if not np.any(Y_ = # TODO
        continue
    plt.scatter(X[Y_ = # TODO

    angle = # TODO
    angle = # TODO
    v = # TODO
    ellipse = # TODO
    ellipse.set_clip_box(fig.bbox)
    ellipse.set_alpha(0.5)
    ax.add_artist(ellipse)

plt.title(
    f"Selected GMM: {grid_search.best_params_['covariance_type']} model, "
    f"{grid_search.best_params_['n_components']} components"
)
plt.axis("equal")
plt.show()
