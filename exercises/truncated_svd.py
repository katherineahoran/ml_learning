import numpy as np
from sklearn.decomposition import TruncatedSVD

# Choose a basis
basis = # TODO

# Generate a sample
n_samples = # TODO

points = # TODO
points_concatenated = # TODO

# Use TruncatedSVD to find a map
svd = # TODO
svd.fit(points)

# Use the map to transform sample and basis
reduced = # TODO
reduced_basis = # TODO

# Look at how well the map preserved integrity
print("reduced points:")
print(reduced)

print("reduced basis:")
print(reduced)

print("reduced points inversed - actual:")
print(svd.inverse_transform(reduced) - points)


print("reduced basis inversed - actual:")
print(svd.inverse_transform(reduced_basis) - basis)
