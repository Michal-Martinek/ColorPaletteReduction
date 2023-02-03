import cv2
import numpy as np
import os

os.chdir(os.path.dirname(__file__))
DEBUG = False

# algorithm ------------------------------
def getDistances(colors, means):
	extended = False
	if len(means.shape) == 2:
		extended = True
		means = means[np.newaxis]
	diff = colors[np.newaxis, :, np.newaxis].astype('int32') - means[:, np.newaxis].astype('int32')
	dists = np.sum(diff ** 2, axis=-1)
	return dists[0] if extended else dists
def adjustClusterCenters(colors, means, k, hyperIterations) -> int:
	dists = getDistances(colors, means)
	clusterIdx = np.argmin(dists, axis=2)

	costs = np.zeros((hyperIterations,))
	for hyperIdx in range(hyperIterations): # TODO: do something about the loops
		for meanIdx in range(k):
			isRelevant = clusterIdx[hyperIdx] == meanIdx
			clusterCoordinates = np.where(isRelevant)[0]
			costs[hyperIdx] += np.where(isRelevant, dists[hyperIdx, :, meanIdx], 0).sum(dtype='int64')
			clusterPoints = colors[clusterCoordinates]
			if clusterPoints.shape[0] == 0:
				continue
			means[hyperIdx, meanIdx] = np.sum(clusterPoints, axis=0) / clusterPoints.shape[0]
	return costs
def initMeans(colors, k, hyperIterations): # TODO: better init of means
	withoutRepeats = np.unique(colors, axis=0)
	assert withoutRepeats.shape[0] > k, 'the k is too big'
	meanIdxs = np.array([np.random.choice(np.arange(withoutRepeats.shape[0]), k, replace=False) for i in range(hyperIterations)])
	return withoutRepeats[meanIdxs]
def doKMeans(colors, k, maxIterations, hyperIterations):
	means = initMeans(colors, k, hyperIterations)
	costHistory = []

	# TODO: remember the best means we have already seen
	for i in range(maxIterations): # TODO: try to guess the optimal stopping point based on the costs
		cost = adjustClusterCenters(colors, means, k, hyperIterations)
		costHistory.append(cost)
	means = means[np.argmin(costHistory[-1])]
	return means, np.array(costHistory)
def applyColorPalette(img, colors, means):
	dists = getDistances(colors, means)
	clusterIdx = np.argmin(dists, axis=1)
	return means[clusterIdx].reshape((img.shape[0], img.shape[1], 3))

# UI ------------------------------------------
def showCostDiagrams(costHistory, hyperIterations): # TODO: join the diagrams into one surface
	for i, historyIdx in enumerate(range(hyperIterations)):
		diag = getCostsDiagram(costHistory[:, historyIdx])
		cv2.imshow(f'CostDiagram_{i}', diag)
def getCostsDiagram(costs):
	desiredHeight = 300
	scale = desiredHeight / max(costs)
	img = np.zeros((desiredHeight, len(costs), 3), dtype='uint8')
	heights = desiredHeight - (np.array(costs) * scale).astype('int32')
	heights = np.minimum(heights, desiredHeight-1)
	img[heights, np.arange(len(costs))] = [255, 255, 255]
	return img
def mouse_click(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print(f'clicked at ({y}, {x})')

def main():
	# k-means params
	k_means = 8
	iterations = 20
	hyperIterations = 5

	img = cv2.imread('bird-original.png')
	colors = img.reshape((-1, 3))
	means, costHistory = doKMeans(colors, k_means, iterations, hyperIterations)
	reducedPalette = applyColorPalette(img, colors, means)
	
	if DEBUG:
		showCostDiagrams(costHistory, hyperIterations)
	cv2.imshow('Bird', img)
	cv2.imshow('Reduced', reducedPalette)
	cv2.setMouseCallback('Bird', mouse_click)
	while cv2.getWindowProperty('Bird', cv2.WND_PROP_VISIBLE) >= 1:
		cv2.waitKey(10)

if __name__ == '__main__':
	main()
