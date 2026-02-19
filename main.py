import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ====== Camera configs ======


def getRWithCurrentCameraPositionAndCameraLookingPoint(
    currentPosition, target=np.array([0.0, 0.0, 0.0])
):
    target = np.array(target, dtype=float)
    currentPosition = np.array(currentPosition, dtype=float)
    forwardDir = target - currentPosition
    forwardDir = forwardDir / np.linalg.norm(forwardDir)
    worldUp = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forwardDir, worldUp)) > 0.99:
        worldUp = np.array([0.0, 0.0, 1.0])
    rightDir = np.cross(worldUp, forwardDir)
    rightDir = rightDir / np.linalg.norm(rightDir)
    upDir = np.cross(forwardDir, rightDir)
    return np.stack([rightDir, upDir, forwardDir], axis=0)


def makePMatrix(rotationMatrix, cameraPosition, calibrationMatrix):
    cameraPosition = np.array(cameraPosition, dtype=np.float64)
    translationVector = -rotationMatrix @ cameraPosition
    rotationAndTranslation = np.hstack(
        [rotationMatrix, translationVector.reshape(3, 1)]
    )
    return calibrationMatrix @ rotationAndTranslation


def project(projectionMatrix, point3D):
    homogeneousPoint3D = np.append(point3D, 1.0)
    homogeneousPoint2D = projectionMatrix @ homogeneousPoint3D
    return homogeneousPoint2D[:2] / homogeneousPoint2D[2]


def getLineOfSight(projectedPoint2D, calibrationMatrix, rotationMatrix, cameraPosition):
    homogeneousPoint2D = np.array([projectedPoint2D[0], projectedPoint2D[1], 1.0])
    rayDirection = (
        rotationMatrix.T @ np.linalg.inv(calibrationMatrix) @ homogeneousPoint2D
    )
    rayDirection = rayDirection / np.linalg.norm(rayDirection)
    return np.array(cameraPosition, dtype=float), rayDirection


def addNoiseToCamera(
    rotationMatrix, cameraPosition, rotationNoiseStd, positionNoiseStd
):
    noisyPosition = cameraPosition + np.random.normal(0, positionNoiseStd, size=3)
    randomAxis = np.random.randn(3)
    randomAxis = randomAxis / np.linalg.norm(randomAxis)
    randomAngle = np.random.normal(0, rotationNoiseStd)
    skewSymmetric = np.array(
        [
            [0, -randomAxis[2], randomAxis[1]],
            [randomAxis[2], 0, -randomAxis[0]],
            [-randomAxis[1], randomAxis[0], 0],
        ]
    )
    rotationNoise = (
        np.eye(3)
        + np.sin(randomAngle) * skewSymmetric
        + (1 - np.cos(randomAngle)) * (skewSymmetric @ skewSymmetric)
    )
    noisyRotation = rotationNoise @ rotationMatrix
    return noisyRotation, noisyPosition


def initCameraConfigs():
    calibrationMatrix = np.array(
        [[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]]
    )
    cameraPositions = {
        1: ([-5.0, -1.0, 0.0], [-5.0, +1.0, 0.0]),
        2: ([-12.0, 0.0, 0.0], [-2.0, 0.0, 0.0]),
        3: ([-10.0, 2.0, -1.0], [-5.0, -2.0, 1.0]),
    }
    cameraConfigs = {}
    for configId, (position1, position2) in cameraPositions.items():
        cameraCenter1 = np.array(position1)
        cameraCenter2 = np.array(position2)
        rotationMatrix1 = getRWithCurrentCameraPositionAndCameraLookingPoint(
            cameraCenter1
        )
        rotationMatrix2 = getRWithCurrentCameraPositionAndCameraLookingPoint(
            cameraCenter2
        )
        projectionMatrix1 = makePMatrix(
            rotationMatrix1, cameraCenter1, calibrationMatrix
        )
        projectionMatrix2 = makePMatrix(
            rotationMatrix2, cameraCenter2, calibrationMatrix
        )
        cameraConfigs[configId] = {
            "P1": projectionMatrix1,
            "R1": rotationMatrix1,
            "c1": cameraCenter1,
            "P2": projectionMatrix2,
            "R2": rotationMatrix2,
            "c2": cameraCenter2,
            "K": calibrationMatrix,
        }
    return cameraConfigs


# ====== Triangulation methods ======


def midpoint(
    projectionMatrix1,
    rotationMatrix1,
    cameraCenter1,
    projectionMatrix2,
    rotationMatrix2,
    cameraCenter2,
    imagePoint1,
    imagePoint2,
    calibrationMatrix,
):
    _, rayDirection1 = getLineOfSight(
        imagePoint1, calibrationMatrix, rotationMatrix1, cameraCenter1
    )
    _, rayDirection2 = getLineOfSight(
        imagePoint2, calibrationMatrix, rotationMatrix2, cameraCenter2
    )
    systemMatrix = np.array(
        [
            [
                np.dot(rayDirection1, rayDirection1),
                -np.dot(rayDirection1, rayDirection2),
            ],
            [
                np.dot(rayDirection1, rayDirection2),
                -np.dot(rayDirection2, rayDirection2),
            ],
        ]
    )
    differenceVector = np.array(
        [
            np.dot(cameraCenter2 - cameraCenter1, rayDirection1),
            np.dot(cameraCenter2 - cameraCenter1, rayDirection2),
        ]
    )
    rayParameters = np.linalg.solve(systemMatrix, differenceVector)
    closestPoint1 = cameraCenter1 + rayParameters[0] * rayDirection1
    closestPoint2 = cameraCenter2 + rayParameters[1] * rayDirection2
    return (closestPoint1 + closestPoint2) / 2


def dltTriangulation(projectionMatrix1, projectionMatrix2, imagePoint1, imagePoint2):
    designMatrix = np.array(
        [
            imagePoint1[0] * projectionMatrix1[2] - projectionMatrix1[0],
            imagePoint1[1] * projectionMatrix1[2] - projectionMatrix1[1],
            imagePoint2[0] * projectionMatrix2[2] - projectionMatrix2[0],
            imagePoint2[1] * projectionMatrix2[2] - projectionMatrix2[1],
        ]
    )
    _, _, rightSingularVectors = np.linalg.svd(designMatrix)
    homogeneousPoint3D = rightSingularVectors[-1]
    return homogeneousPoint3D[:3] / homogeneousPoint3D[3]


def computeFundamentalMatrix(projectionMatrix1, projectionMatrix2):
    _, _, rightSingularVectors = np.linalg.svd(projectionMatrix1)
    nullSpacePoint = rightSingularVectors[-1]
    cameraCenter1 = nullSpacePoint[:3] / nullSpacePoint[3]
    epipole2 = projectionMatrix2 @ np.append(cameraCenter1, 1.0)
    epipole2Skew = np.array(
        [
            [0, -epipole2[2], epipole2[1]],
            [epipole2[2], 0, -epipole2[0]],
            [-epipole2[1], epipole2[0], 0],
        ]
    )
    return epipole2Skew @ projectionMatrix2 @ np.linalg.pinv(projectionMatrix1)


def normalizePointsForL1(homogeneousPoint1, homogeneousPoint2, fundamentalMatrix):
    translationToOrigin1 = np.array(
        [[1, 0, -homogeneousPoint1[0]], [0, 1, -homogeneousPoint1[1]], [0, 0, 1]]
    )
    translationToOrigin2 = np.array(
        [[1, 0, -homogeneousPoint2[0]], [0, 1, -homogeneousPoint2[1]], [0, 0, 1]]
    )
    translatedF = (
        np.linalg.inv(translationToOrigin2).T
        @ fundamentalMatrix
        @ np.linalg.inv(translationToOrigin1)
    )

    _, _, rightVectors1 = np.linalg.svd(translatedF)
    epipole1 = rightVectors1[-1]
    epipole1 = epipole1 / epipole1[2]

    _, _, rightVectors2 = np.linalg.svd(translatedF.T)
    epipole2 = rightVectors2[-1]
    epipole2 = epipole2 / epipole2[2]

    epipoleNorm1 = np.sqrt(epipole1[0] ** 2 + epipole1[1] ** 2)
    epipoleNorm2 = np.sqrt(epipole2[0] ** 2 + epipole2[1] ** 2)

    rotationForEpipole1 = np.array(
        [
            [epipole1[0] / epipoleNorm1, epipole1[1] / epipoleNorm1, 0],
            [-epipole1[1] / epipoleNorm1, epipole1[0] / epipoleNorm1, 0],
            [0, 0, 1],
        ]
    )
    rotationForEpipole2 = np.array(
        [
            [epipole2[0] / epipoleNorm2, epipole2[1] / epipoleNorm2, 0],
            [-epipole2[1] / epipoleNorm2, epipole2[0] / epipoleNorm2, 0],
            [0, 0, 1],
        ]
    )

    focalLength1 = epipole1[2] / epipoleNorm1
    focalLength2 = epipole2[2] / epipoleNorm2
    normalizedF = rotationForEpipole2 @ translatedF @ rotationForEpipole1.T
    fullTransform1 = rotationForEpipole1 @ translationToOrigin1
    fullTransform2 = rotationForEpipole2 @ translationToOrigin2

    return fullTransform1, fullTransform2, focalLength1, focalLength2, normalizedF


def buildL1Polynomial(a, b, c, d, focalLength, focalLengthPrime):
    """
    چندجمله‌ای درجه 8 — Hartley & Sturm 1997 Section 5.4
    g1(t) = (at+b)^2 + f'^2*(ct+d)^2
    g2(t) = 1 + f^2*t^2
    شرط: g1(t)^3 = (ad-bc)^2 * (at+b)^2 * g2(t)^3
    """
    g1Coeffs = np.array(
        [
            a**2 + focalLengthPrime**2 * c**2,
            2 * (a * b + focalLengthPrime**2 * c * d),
            b**2 + focalLengthPrime**2 * d**2,
        ]
    )
    g2Coeffs = np.array([focalLength**2, 0.0, 1.0])
    linearTermSq = np.array([a**2, 2 * a * b, b**2])
    crossTermDet = a * d - b * c

    g1Cubed = np.polymul(np.polymul(g1Coeffs, g1Coeffs), g1Coeffs)
    g2Cubed = np.polymul(np.polymul(g2Coeffs, g2Coeffs), g2Coeffs)
    rightHandSide = (crossTermDet**2) * np.polymul(linearTermSq, g2Cubed)
    leftHandSide = np.concatenate([[0, 0], g1Cubed])
    return rightHandSide - leftHandSide


def computeL1ReprojectionCost(t, a, b, c, d, focalLength, focalLengthPrime):
    denominator1 = np.sqrt(1 + (t * focalLength) ** 2)
    denominator2 = np.sqrt((a * t + b) ** 2 + focalLengthPrime**2 * (c * t + d) ** 2)
    if denominator1 < 1e-10 or denominator2 < 1e-10:
        return np.inf
    return abs(t) / denominator1 + abs(c * t + d) / denominator2


def l1(
    projectionMatrix1,
    rotationMatrix1,
    cameraCenter1,
    projectionMatrix2,
    rotationMatrix2,
    cameraCenter2,
    imagePoint1,
    imagePoint2,
    calibrationMatrix,
):
    """
    L1 Triangulation — Hartley & Sturm 1997 Section 5.4
    حل closed-form با چندجمله‌ای درجه 8
    """
    homogeneousPoint1 = np.array([imagePoint1[0], imagePoint1[1], 1.0])
    homogeneousPoint2 = np.array([imagePoint2[0], imagePoint2[1], 1.0])

    fundamentalMatrix = computeFundamentalMatrix(projectionMatrix1, projectionMatrix2)
    transform1, transform2, focalLength, focalLengthPrime, normalizedF = (
        normalizePointsForL1(homogeneousPoint1, homogeneousPoint2, fundamentalMatrix)
    )

    a = normalizedF[1, 1]
    b = normalizedF[1, 2]
    c = normalizedF[2, 1]
    d = normalizedF[2, 2]

    polynomialCoefficients = buildL1Polynomial(
        a, b, c, d, focalLength, focalLengthPrime
    )
    allRoots = np.roots(polynomialCoefficients)
    realRoots = [np.real(root) for root in allRoots if abs(np.imag(root)) < 1e-6]
    tCandidates = realRoots + [1e10]

    bestT = min(
        tCandidates,
        key=lambda t: computeL1ReprojectionCost(
            t, a, b, c, d, focalLength, focalLengthPrime
        ),
    )

    # بازسازی نقطه تصحیح‌شده در فضای نرمال
    correctedPoint1Normalized = np.array([bestT, 0.0, 1.0])

    epilineInImage2 = normalizedF @ correctedPoint1Normalized
    lineA = epilineInImage2[0]
    lineB = epilineInImage2[1]
    lineC = epilineInImage2[2]
    lineDenominator = lineA**2 + lineB**2
    if lineDenominator < 1e-10:
        return midpoint(
            projectionMatrix1,
            rotationMatrix1,
            cameraCenter1,
            projectionMatrix2,
            rotationMatrix2,
            cameraCenter2,
            imagePoint1,
            imagePoint2,
            calibrationMatrix,
        )

    # نزدیک‌ترین نقطه به مبدأ روی خط اپی‌پولار (اصلاح‌شده)
    correctedPoint2Normalized = np.array(
        [-lineA * lineC / lineDenominator, -lineB * lineC / lineDenominator, 1.0]
    )

    # برگشت به فضای اصلی با inverse transform
    correctedPoint1Original = np.linalg.inv(transform1) @ correctedPoint1Normalized
    correctedPoint2Original = np.linalg.inv(transform2) @ correctedPoint2Normalized

    correctedImagePoint1 = correctedPoint1Original[:2] / correctedPoint1Original[2]
    correctedImagePoint2 = correctedPoint2Original[:2] / correctedPoint2Original[2]

    return dltTriangulation(
        projectionMatrix1, projectionMatrix2, correctedImagePoint1, correctedImagePoint2
    )


def anglL2(
    projectionMatrix1,
    rotationMatrix1,
    cameraCenter1,
    projectionMatrix2,
    rotationMatrix2,
    cameraCenter2,
    imagePoint1,
    imagePoint2,
    calibrationMatrix,
):
    """Angular L2 — Lee & Civera 2019"""
    _, rayDirection1 = getLineOfSight(
        imagePoint1, calibrationMatrix, rotationMatrix1, cameraCenter1
    )
    _, rayDirection2 = getLineOfSight(
        imagePoint2, calibrationMatrix, rotationMatrix2, cameraCenter2
    )

    def angularL2Cost(point3D):
        vectorToPoint1 = point3D - cameraCenter1
        vectorToPoint2 = point3D - cameraCenter2
        norm1 = np.linalg.norm(vectorToPoint1)
        norm2 = np.linalg.norm(vectorToPoint2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 1e10
        angle1 = np.arccos(
            np.clip(np.dot(vectorToPoint1 / norm1, rayDirection1), -1, 1)
        )
        angle2 = np.arccos(
            np.clip(np.dot(vectorToPoint2 / norm2, rayDirection2), -1, 1)
        )
        return angle1**2 + angle2**2

    initialPoint = midpoint(
        projectionMatrix1,
        rotationMatrix1,
        cameraCenter1,
        projectionMatrix2,
        rotationMatrix2,
        cameraCenter2,
        imagePoint1,
        imagePoint2,
        calibrationMatrix,
    )
    return minimize(
        angularL2Cost,
        initialPoint,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 10000},
    ).x


def anglL1(
    projectionMatrix1,
    rotationMatrix1,
    cameraCenter1,
    projectionMatrix2,
    rotationMatrix2,
    cameraCenter2,
    imagePoint1,
    imagePoint2,
    calibrationMatrix,
):
    """Angular L1 — Lee & Civera 2019"""
    _, rayDirection1 = getLineOfSight(
        imagePoint1, calibrationMatrix, rotationMatrix1, cameraCenter1
    )
    _, rayDirection2 = getLineOfSight(
        imagePoint2, calibrationMatrix, rotationMatrix2, cameraCenter2
    )

    def angularL1Cost(point3D):
        vectorToPoint1 = point3D - cameraCenter1
        vectorToPoint2 = point3D - cameraCenter2
        norm1 = np.linalg.norm(vectorToPoint1)
        norm2 = np.linalg.norm(vectorToPoint2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 1e10
        angle1 = np.arccos(
            np.clip(np.dot(vectorToPoint1 / norm1, rayDirection1), -1, 1)
        )
        angle2 = np.arccos(
            np.clip(np.dot(vectorToPoint2 / norm2, rayDirection2), -1, 1)
        )
        return abs(angle1) + abs(angle2)

    initialPoint = midpoint(
        projectionMatrix1,
        rotationMatrix1,
        cameraCenter1,
        projectionMatrix2,
        rotationMatrix2,
        cameraCenter2,
        imagePoint1,
        imagePoint2,
        calibrationMatrix,
    )
    return minimize(
        angularL1Cost,
        initialPoint,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 10000},
    ).x


def runTriangulation(
    methodName,
    projectionMatrix1,
    rotationMatrix1,
    cameraCenter1,
    projectionMatrix2,
    rotationMatrix2,
    cameraCenter2,
    imagePoint1,
    imagePoint2,
    calibrationMatrix,
):
    args = (
        projectionMatrix1,
        rotationMatrix1,
        cameraCenter1,
        projectionMatrix2,
        rotationMatrix2,
        cameraCenter2,
        imagePoint1,
        imagePoint2,
        calibrationMatrix,
    )
    if methodName == "MP":
        return midpoint(*args)
    elif methodName == "L1":
        return l1(*args)
    elif methodName == "AngL2":
        return anglL2(*args)
    elif methodName == "AngL1":
        return anglL1(*args)


# ====== Sensitivity Experiments ======


def makeSamplePointInSphere(radius=0.25):
    azimuthAngle = np.random.uniform(0, 2 * np.pi)
    cosElevation = np.random.uniform(-1, 1)
    uniformSample = np.random.uniform(0, 1)
    elevationAngle = np.arccos(cosElevation)
    radialDistance = radius * (uniformSample ** (1 / 3))
    x = radialDistance * np.sin(elevationAngle) * np.cos(azimuthAngle)
    y = radialDistance * np.sin(elevationAngle) * np.sin(azimuthAngle)
    z = radialDistance * np.cos(elevationAngle)
    return np.array([x, y, z])


def computeAngleBetweenVectors(vector1, vector2):
    cosAngle = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    return np.arccos(np.clip(cosAngle, -1.0, 1.0))


# ====== Sensitivity methods ======


def positionErrorSensitivity(configId, methods, nIters=100, noiseLevels=range(1, 11)):
    cameraConfigs = initCameraConfigs()
    activeCam = cameraConfigs[configId]
    results = {method: [] for method in methods}

    for noiseLevel in noiseLevels:
        positionNoiseStd = 0.01 * noiseLevel
        angleNoiseStd = np.deg2rad(0.1 * noiseLevel)
        iterationErrors = {method: [] for method in methods}

        for _ in range(nIters):
            truePoint3D = makeSamplePointInSphere(radius=0.25)
            imagePoint1 = project(activeCam["P1"], truePoint3D)
            imagePoint2 = project(activeCam["P2"], truePoint3D)

            noisyRotation1, noisyCenter1 = addNoiseToCamera(
                activeCam["R1"], activeCam["c1"], angleNoiseStd, positionNoiseStd
            )
            noisyRotation2, noisyCenter2 = addNoiseToCamera(
                activeCam["R2"], activeCam["c2"], angleNoiseStd, positionNoiseStd
            )
            noisyProjection1 = makePMatrix(noisyRotation1, noisyCenter1, activeCam["K"])
            noisyProjection2 = makePMatrix(noisyRotation2, noisyCenter2, activeCam["K"])

            for method in methods:
                estimatedPoint3D = runTriangulation(
                    method,
                    noisyProjection1,
                    noisyRotation1,
                    noisyCenter1,
                    noisyProjection2,
                    noisyRotation2,
                    noisyCenter2,
                    imagePoint1,
                    imagePoint2,
                    activeCam["K"],
                )
                iterationErrors[method].append(
                    np.linalg.norm(estimatedPoint3D - truePoint3D)
                )

        for method in methods:
            results[method].append(np.mean(iterationErrors[method]))
    return results


def distanceErrorSensitivity(configId, methods, nIters=100, noiseLevels=range(1, 11)):
    cameraConfigs = initCameraConfigs()
    activeCam = cameraConfigs[configId]
    results = {method: [] for method in methods}

    for noiseLevel in noiseLevels:
        positionNoiseStd = 0.01 * noiseLevel
        angleNoiseStd = np.deg2rad(0.1 * noiseLevel)
        iterationErrors = {method: [] for method in methods}

        for _ in range(nIters):
            truePoint1 = makeSamplePointInSphere(radius=0.25)
            truePoint2 = makeSamplePointInSphere(radius=0.25)

            imagePoint1ofPoint1 = project(activeCam["P1"], truePoint1)
            imagePoint2ofPoint1 = project(activeCam["P2"], truePoint1)
            imagePoint1ofPoint2 = project(activeCam["P1"], truePoint2)
            imagePoint2ofPoint2 = project(activeCam["P2"], truePoint2)

            noisyRotation1, noisyCenter1 = addNoiseToCamera(
                activeCam["R1"], activeCam["c1"], angleNoiseStd, positionNoiseStd
            )
            noisyRotation2, noisyCenter2 = addNoiseToCamera(
                activeCam["R2"], activeCam["c2"], angleNoiseStd, positionNoiseStd
            )
            noisyProjection1 = makePMatrix(noisyRotation1, noisyCenter1, activeCam["K"])
            noisyProjection2 = makePMatrix(noisyRotation2, noisyCenter2, activeCam["K"])

            for method in methods:
                estimatedPoint1 = runTriangulation(
                    method,
                    noisyProjection1,
                    noisyRotation1,
                    noisyCenter1,
                    noisyProjection2,
                    noisyRotation2,
                    noisyCenter2,
                    imagePoint1ofPoint1,
                    imagePoint2ofPoint1,
                    activeCam["K"],
                )
                estimatedPoint2 = runTriangulation(
                    method,
                    noisyProjection1,
                    noisyRotation1,
                    noisyCenter1,
                    noisyProjection2,
                    noisyRotation2,
                    noisyCenter2,
                    imagePoint1ofPoint2,
                    imagePoint2ofPoint2,
                    activeCam["K"],
                )
                trueDistance = np.linalg.norm(truePoint1 - truePoint2)
                estimatedDistance = np.linalg.norm(estimatedPoint1 - estimatedPoint2)
                iterationErrors[method].append(abs(trueDistance - estimatedDistance))

        for method in methods:
            results[method].append(np.mean(iterationErrors[method]))
    return results


def angleErrorSensitivity(configId, methods, nIters=100, noiseLevels=range(1, 11)):
    cameraConfigs = initCameraConfigs()
    activeCam = cameraConfigs[configId]
    results = {method: [] for method in methods}

    for noiseLevel in noiseLevels:
        positionNoiseStd = 0.01 * noiseLevel
        angleNoiseStd = np.deg2rad(0.1 * noiseLevel)
        iterationErrors = {method: [] for method in methods}

        for _ in range(nIters):
            truePoint1 = makeSamplePointInSphere(radius=0.25)
            truePoint2 = makeSamplePointInSphere(radius=0.25)
            truePoint3 = makeSamplePointInSphere(radius=0.25)

            noisyRotation1, noisyCenter1 = addNoiseToCamera(
                activeCam["R1"], activeCam["c1"], angleNoiseStd, positionNoiseStd
            )
            noisyRotation2, noisyCenter2 = addNoiseToCamera(
                activeCam["R2"], activeCam["c2"], angleNoiseStd, positionNoiseStd
            )
            noisyProjection1 = makePMatrix(noisyRotation1, noisyCenter1, activeCam["K"])
            noisyProjection2 = makePMatrix(noisyRotation2, noisyCenter2, activeCam["K"])

            for method in methods:
                estimatedPoint1 = runTriangulation(
                    method,
                    noisyProjection1,
                    noisyRotation1,
                    noisyCenter1,
                    noisyProjection2,
                    noisyRotation2,
                    noisyCenter2,
                    project(activeCam["P1"], truePoint1),
                    project(activeCam["P2"], truePoint1),
                    activeCam["K"],
                )
                estimatedPoint2 = runTriangulation(
                    method,
                    noisyProjection1,
                    noisyRotation1,
                    noisyCenter1,
                    noisyProjection2,
                    noisyRotation2,
                    noisyCenter2,
                    project(activeCam["P1"], truePoint2),
                    project(activeCam["P2"], truePoint2),
                    activeCam["K"],
                )
                estimatedPoint3 = runTriangulation(
                    method,
                    noisyProjection1,
                    noisyRotation1,
                    noisyCenter1,
                    noisyProjection2,
                    noisyRotation2,
                    noisyCenter2,
                    project(activeCam["P1"], truePoint3),
                    project(activeCam["P2"], truePoint3),
                    activeCam["K"],
                )
                trueAngle = computeAngleBetweenVectors(
                    truePoint2 - truePoint1, truePoint3 - truePoint1
                )
                estimatedAngle = computeAngleBetweenVectors(
                    estimatedPoint2 - estimatedPoint1, estimatedPoint3 - estimatedPoint1
                )
                iterationErrors[method].append(abs(trueAngle - estimatedAngle))

        for method in methods:
            results[method].append(np.mean(iterationErrors[method]))
    return results


# ====== Experiment statistics ======


def computeStats(allRunsResults):
    statsPerMethod = {}
    for methodName, errorValues in allRunsResults.items():
        errorArray = np.array(errorValues)
        statsPerMethod[methodName] = {
            "Mean": np.mean(errorArray),
            "Median": np.median(errorArray),
            "Std": np.std(errorArray),
            "Min": np.min(errorArray),
            "Max": np.max(errorArray),
        }
    return pd.DataFrame(statsPerMethod).T


def runFullExperiment(
    configId, methods, nRuns=100, nIters=20, noiseLevels=range(1, 11)
):
    allRunsMeanErrors = {method: [] for method in methods}
    for runIndex in range(nRuns):
        if runIndex % 10 == 0:
            print(f"  Run {runIndex}/{nRuns}...")
        sensitivityResults = positionErrorSensitivity(
            configId, methods, nIters=nIters, noiseLevels=noiseLevels
        )
        for method in methods:
            allRunsMeanErrors[method].append(np.mean(sensitivityResults[method]))
    return allRunsMeanErrors


def printStatsTable(statsDataFrame, testName, configId):
    print(f"\n{'='*60}")
    print(f"  {testName} Error — Configuration {configId}")
    print(f"{'='*60}")
    print(statsDataFrame.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"{'='*60}\n")


# ====== Visualization ======


def plotSensitivity(allConfigResults, testName, noiseLevels=range(1, 11)):
    markerStyles = {"MP": "o-", "L1": "s--", "AngL2": "^-.", "AngL1": "x:"}
    figure, axes = plt.subplots(3, 1, figsize=(8, 12))
    figure.suptitle(f"{testName} Error Sensitivity", fontsize=14)
    for subplotIndex, configId in enumerate([1, 2, 3]):
        currentAxis = axes[subplotIndex]
        for methodName, errorValues in allConfigResults[configId].items():
            currentAxis.plot(
                list(noiseLevels),
                errorValues,
                markerStyles.get(methodName, "o-"),
                label=methodName,
                linewidth=1.5,
            )
        currentAxis.set_title(f"Configuration {configId}")
        currentAxis.set_xlabel("Noise Level")
        currentAxis.set_ylabel("Mean Error")
        currentAxis.legend()
        currentAxis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figure_{testName.lower()}.pdf")
    plt.show()


# ====== Main ======

if __name__ == "__main__":
    methodNames = ["MP", "L1", "AngL2", "AngL1"]
    noiseLevels = range(1, 11)

    # بخش ۱: نمودارهای Sensitivity
    for testName, sensitivityFunction in [
        ("Position", positionErrorSensitivity),
        ("Distance", distanceErrorSensitivity),
        ("Angle", angleErrorSensitivity),
    ]:
        allConfigResults = {}
        for configId in [1, 2, 3]:
            print(f"Running {testName} - Config {configId}...")
            allConfigResults[configId] = sensitivityFunction(
                configId, methodNames, nIters=10, noiseLevels=noiseLevels
            )
        plotSensitivity(allConfigResults, testName, noiseLevels)

    # بخش ۲: جداول آماری (Table 1 و Table 2 مقاله)
    print("\nComputing statistics tables...")
    for configId in [1, 2, 3]:
        print(f"\nConfig {configId}:")
        allRunsMeanErrors = runFullExperiment(
            configId, methodNames, nRuns=100, nIters=20
        )
        statsTable = computeStats(allRunsMeanErrors)
        printStatsTable(statsTable, "Position", configId)
