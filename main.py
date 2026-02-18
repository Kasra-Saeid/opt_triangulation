# ====== Camera configs ======
def getRWithCurrentCameraPositionAndCameraLookingPoint(currentPosition, target):
    pass


def makePMatrix(r, c, k):
    pass


def project(pMatrix, point3D):
    pass


def getLineOfSight(u, k, r, c):
    pass


def addNoiseToCamera(r, c, stdR, stdC):
    pass


def initCameraConfigs():
    pass


# ====== Optimization methods ======
def l1():
    pass


def anglL1():
    pass


def anglL2():
    pass


# ====== Sensitivity methods ======
def positionErrorSensitivity():
    pass


def distanceErrorSensitivity():
    pass


def angleErrorSensitivity():
    pass


# ====== Sensitivity Experiments ======
def makeSamplePointInSphere(radius=0.25):  #! why this formula
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    u = np.random.uniform(0, 1)
    theta = np.arccos(costheta)
    r = radius * (u ** (1 / 3))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


# ====== Experiment statistics ======


# ====== Visualization ======


if __name__ == "__main__":
    pass
