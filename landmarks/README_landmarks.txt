# Code to load the landmarks:
# the 5 columns are X, Y, Z, u, v

pts_loaded = np.loadtxt("landmarks/parking.txt", delimiter=",")
points3D, points2D = pts_loaded[:,:3], pts_loaded[:,3:]