import cv2
import numpy as np
import os

os.chdir(os.path.dirname(__file__))
DEBUG = False

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
def initMeans(colors, k): # TODO: better means init
	withoutRepeats = np.unique(colors, axis=0)
	assert withoutRepeats.shape[0] > k, 'the k is too big'
	meanIdxs = np.random.choice(np.arange(withoutRepeats.shape[0]), k, replace=False)
	return withoutRepeats[meanIdxs]
def doKMeans(colors, k, maxIterations, autoStop=not DEBUG):
	means = initMeans(colors, k)
	lastCosts = []

	lastCost = -1
	bestMeans = None
	bestCost = float('inf')
	for i in range(maxIterations): # TODO: try to guess the optimal stopping point based on the costs
		cost = adjustClusterCenters(colors, means, k)
		lastCosts.append(cost)
		if cost < bestCost:
			bestCost = cost
			bestMeans = means.copy()
		if autoStop and cost == lastCost:
			break
		lastCost = cost
	return bestMeans, lastCosts
def applyColorPalette(img, colors, means):
	dists = getDistances(colors, means)
	clusterIdx = np.argmin(dists, axis=1)
	return means[clusterIdx].reshape((img.shape[0], img.shape[1], 3))
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
	img = cv2.imread('bird-original.png')
	colors = img.reshape((-1, 3))
	means, costs = doKMeans(colors, 8, 50)
	reducedPalette = applyColorPalette(img, colors, means)
	
	cv2.imshow('Bird', img)
	cv2.imshow('Reduced', reducedPalette)
	if DEBUG:
		diagram = getCostsDiagram(costs)
		cv2.imshow('CostGraph', diagram)
	cv2.setMouseCallback('Bird', mouse_click)
	while cv2.getWindowProperty('Bird', cv2.WND_PROP_VISIBLE) >= 1:
		cv2.waitKey(10)

if __name__ == '__main__':
	main()
