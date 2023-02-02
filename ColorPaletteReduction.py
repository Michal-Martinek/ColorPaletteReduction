import cv2
import numpy as np
import os

os.chdir(os.path.dirname(__file__))

def getDistances(colors, means):
	diff = colors[:, np.newaxis].astype('int32') - means[np.newaxis].astype('int32')
	return np.sum(diff ** 2, axis=-1)
def adjustClusterCenters(colors, means, k) -> int:
	dists = getDistances(colors, means)
	clusterIdx = np.argmin(dists, axis=1)

	for meanIdx in range(k):
		clusterPoints = colors[np.where(clusterIdx == meanIdx)[0]]
		if clusterPoints.shape[0] == 0:
			continue
		means[meanIdx] = np.sum(clusterPoints, axis=0) / clusterPoints.shape[0]
	return sum([np.where(clusterIdx == meanIdx, dists[:, meanIdx], 0).sum(dtype='int64') for meanIdx in range(k)])
def initMeans(colors, k):
	withoutRepeats = np.unique(colors, axis=0)
	assert withoutRepeats.shape[0] > k, 'the k is too big'
	meanIdxs = np.random.choice(np.arange(withoutRepeats.shape[0]), k, replace=False)
	return withoutRepeats[meanIdxs]
def doKMeans(colors, k, maxIterations):
	means = initMeans(colors, k)
	lastCost = -1
	bestMeans = None
	bestCost = float('inf')
	for i in range(maxIterations):
		cost = adjustClusterCenters(colors, means, k)
		print(cost)

		if cost < bestCost:
			bestCost = cost
			bestMeans = means.copy()
		if cost == lastCost:
			break
		lastCost = cost
	return bestMeans

def mouse_click(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print(f'clicked at ({y}, {x})')
def main():
	img = cv2.imread('bird-original.png')
	colors = img.reshape((-1, 3))
	means = doKMeans(colors, 3, 10)
	
	dists = getDistances(colors, means)
	clusterIdx = np.argmin(dists, axis=1)
	reducedPalette = means[clusterIdx].reshape((img.shape[0], img.shape[1], 3))
	
	cv2.imshow('Bird', img)
	cv2.imshow('Reduced', reducedPalette)
	cv2.setMouseCallback('Bird', mouse_click)
	while cv2.getWindowProperty('Bird', cv2.WND_PROP_VISIBLE) >= 1:
		cv2.waitKey(10)

if __name__ == '__main__':
	main()
