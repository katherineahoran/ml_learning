import numpy as np
from sklearn.decomposition import TruncatedSVD

basis = np.array([ [1 , 1, 0, 0], [0, 1, 3, 0] ])

n_samples = 120

error_scaling = 10**(-20)

points = np.array([ x[0] * basis[0] + x[1] * basis[1] for x in np.random.randn(n_samples, 2) ])
points_concatenated = np.concatenate(points)

svd = TruncatedSVD(random_state=1, n_components=2)
svd.fit(points)

reduced = svd.transform(points)

reduced_basis = svd.transform(basis)

print("reduced points:")
print(reduced)

print("reduced basis:")
print(reduced_basis)

print("reduced points inversed - actual:")
print(svd.inverse_transform(reduced) - points)


print("reduced basis inversed - actual:")
print(svd.inverse_transform(reduced_basis) - basis)
