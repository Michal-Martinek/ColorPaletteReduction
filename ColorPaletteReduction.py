import cv2
import numpy as np
import time
import os

os.chdir(os.path.dirname(__file__))
DEBUG = True
SAVE = True
MAX_MEGS_FOR_DISTS = 200

# algorithm ------------------------------
def initDists(shape):
	expectedSize = shape[0] * shape[1] * shape[2] * 4 / 1_000_000
	if expectedSize > MAX_MEGS_FOR_DISTS:
		raise RuntimeError(f'Expected size {expectedSize:.1f} MB of the dists array would be probably too large,\n\tchange the MAX_MEGS_FOR_DISTS const respectively of your hardware')
	dists = np.empty(shape, dtype='int32')
	return dists
def getDistances(colors, means, dists=None):
	extended = False
	if len(means.shape) == 2:
		extended = True
		means = means[np.newaxis]
	if dists is None:
		dists = initDists((means.shape[0], colors.shape[0], means.shape[1]))
	
	np.sum((colors[np.newaxis, :, np.newaxis].astype('int32') - means[:, np.newaxis].astype('int32')) ** 2, axis=-1, out=dists)
	return dists[0] if extended else dists
def adjustClusterCenters(colors, means, k, hyperIterations, dists=None):
	dists = getDistances(colors, means, dists)
	clusterIdx = np.argmin(dists, axis=2)

	costs = np.zeros((hyperIterations,), dtype='int64')
	for hyperIdx in range(hyperIterations): # TODO: do something about the loops
		for meanIdx in range(k):
			isRelevant = clusterIdx[hyperIdx] == meanIdx
			clusterCoordinates = np.where(isRelevant)[0]
			costs[hyperIdx] += np.where(isRelevant, dists[hyperIdx, :, meanIdx], 0).sum(dtype='int64')
			clusterPoints = colors[clusterCoordinates]
			if clusterPoints.shape[0] == 0:
				continue
			means[hyperIdx, meanIdx] = np.sum(clusterPoints, axis=0) / clusterPoints.shape[0]
	return costs, dists
def initMeans(colors, k, hyperIterations): # TODO: better init of means, maybe pick ony ones with low cost
	withoutRepeats = np.unique(colors, axis=0)
	assert withoutRepeats.shape[0] > k, 'the k is too big'
	meanIdxs = np.array([np.random.choice(np.arange(withoutRepeats.shape[0]), k, replace=False) for i in range(hyperIterations)])
	return withoutRepeats[meanIdxs]
def doKMeans(colors, k, maxIterations, hyperIterations):
	means = initMeans(colors, k, hyperIterations)
	costHistory = []
	dists = None

	# TODO: remember the best means we have already seen
	for i in range(maxIterations): # TODO: try to guess the optimal stopping point based on the costs
		startTime = time.time()
		cost, dists = adjustClusterCenters(colors, means, k, hyperIterations, dists)
		costHistory.append(cost)
		if DEBUG:
			duration = (time.time() - startTime) * 1000
			print(f'Iteration {str(i).rjust(len(str(maxIterations)))} took {duration:.2f} ms')
	means = means[np.argmin(costHistory[-1])]
	return means, np.array(costHistory)
def applyColorPalette(img, colors, means):
	dists = getDistances(colors, means)
	clusterIdx = np.argmin(dists, axis=1)
	return means[clusterIdx].reshape((img.shape[0], img.shape[1], 3))

# UI ------------------------------------------
def showCostDiagrams(costHistory, hyperIterations, startTimestep=2, height=600, widthScale=3, boundaries=0.05):
	assert startTimestep < costHistory.shape[0], 'invalid startTimestep for given costHistory'
	assert costHistory.shape[0] > startTimestep + 1, 'graph can\'t be shown for one iteration'
	diagWidth = (costHistory.shape[0] - startTimestep) * widthScale + 1
	# map the ranges (max, min) -> ((1-boundaries) * height, bot * height)
	minCost = np.min(costHistory[startTimestep:])
	dataSpread = np.max(costHistory[startTimestep:]) - minCost
	scalingFactor = (1. - 2*boundaries) * height / dataSpread
	img = np.zeros((height, hyperIterations * diagWidth, 3), dtype='uint8')
	for historyIdx in range(hyperIterations):
		heights = scalingFactor * (costHistory[startTimestep:, historyIdx] - minCost) + boundaries * height
		heights = height - heights.astype('int32')
		cols = np.arange(historyIdx * diagWidth, (historyIdx+1)*diagWidth - 1, widthScale)
		for width in range(widthScale):
			img[heights, cols + width] = [255, 255, 255]
		img[:, (historyIdx+1) * diagWidth - 1] = [255, 255, 255]
	cv2.imshow(f'CostDiagram', img)

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
	
	if SAVE:
		cv2.imwrite('saved.png', reducedPalette)
	if DEBUG:
		showCostDiagrams(costHistory, hyperIterations)
	cv2.imshow('Bird', img)
	cv2.imshow('Reduced', reducedPalette)
	cv2.setMouseCallback('Bird', mouse_click)
	while cv2.getWindowProperty('Bird', cv2.WND_PROP_VISIBLE) >= 1:
		cv2.waitKey(10)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
