package learnedsort

import (
	"math"
	"math/rand"
	"sort"
)

// TODO use vectorization
// TODO profile and optimize memory

type LearnedSortingConfig struct {
	ModelCount     []int
	BucketsCount   int
	BucketCapacity int
	BucketsSizes   []int
	Threshold      int
	InputSize      int
	FanOut         int
}

type LinearModel struct {
	Slope     float64
	Intercept float64
}

type RMIModel [][]LinearModel

// LearnedSort sorts your array based on LearnedSort algorithm and using the given model.
func LearnedSort(arr []float64, model RMIModel, fanOut, threshold int, modelCount []int) []float64 {
	// START of preparation

	inputSize := len(arr)
	spillBucket := make([]float64, 0, len(arr)/2)
	readArr := arr

	config := LearnedSortingConfig{
		ModelCount:     modelCount,
		BucketsCount:   fanOut,
		BucketCapacity: inputSize / fanOut,
		BucketsSizes:   nil,
		Threshold:      threshold,
		InputSize:      inputSize,
		FanOut:         fanOut,
	}

	// we suppose these buckets exist and are already filled,
	// because we use them to iterate the readArr in bucketize()
	// I could've implement it without it, but it would require extra logic.
	config.BucketsSizes = make([]int, config.BucketsCount)
	for i := range config.BucketsSizes {
		config.BucketsSizes[i] = config.BucketCapacity
	}

	// we want even sized writeArr and readArr (because we swap them in bucketize),
	// so we add the extra elements to spill.
	spillBucket = append(spillBucket, readArr[config.BucketsCount*config.BucketCapacity:]...)
	readArr = readArr[:config.BucketsCount*config.BucketCapacity]
	config.InputSize = len(readArr)
	writeArr := make([]float64, config.BucketsCount*config.BucketCapacity)

	inferCache := make(map[float64]float64)

	// END of preparation

	// Transform array to buckets
	readArr = bucketize(readArr, writeArr, &spillBucket, model, &config, inferCache)

	// Reorder each bucket (makes the sorting slower)
	// reorderBucketElements(readArr, model, &config, inferCache)

	// remove unfilled parts of buckets
	readArr = compact(readArr, &config)

	// Fix potential collisions
	insertionSort(readArr)

	// Sort and merge spill and sorted array
	sort.Float64s(spillBucket)
	return merge(readArr, spillBucket)
}

func bucketize(readArr, writeArr []float64, spillBucket *[]float64, model RMIModel, config *LearnedSortingConfig, inferCache map[float64]float64) []float64 {
	/*
		At the end of each loop, we swap readArr and writeArr.
		Because of this, unwanted values may remain in the array we're working with.
		So instead of directly iterating readArr, we iterate buckets.

		We iterate current buckets on readArr, to create new buckets in writeArr.
	*/
	currentBucketCapacity := config.BucketCapacity
	for {
		// deep copy
		currentBucketsCount := len(config.BucketsSizes)
		currentBucketsSizes := make([]int, currentBucketsCount)
		copy(currentBucketsSizes, config.BucketsSizes)

		config.BucketsSizes = make([]int, config.BucketsCount)

		modelCount := config.ModelCount
		nextBucketsCount := config.BucketsCount
		nextBucketsSizes := config.BucketsSizes
		nextBucketCapacity := config.BucketCapacity

		offset := 0
		for i := 0; i < currentBucketsCount; i++ {
			// record the counts of the predicted position
			for j := 0; j < currentBucketsSizes[i]; j++ {
				element := readArr[offset+j]

				// Determine bucket position
				// for future development: caching sometimes throws error, and slows down
				// var pos int
				// if cachedInfer, found := inferCache[element]; found {
				// 	pos = int(cachedInfer * float64(nextBucketsCount))
				// } else {
				// 	// must calculate in [0, bucketsCount) range
				// 	inferResult := infer(model, element, modelCount)
				// 	pos = int(inferResult * float64(nextBucketsCount))
				// 	inferCache[element] = inferResult
				// }
				// must calculate in [0, bucketsCount) range
				pos := int(infer(model, element, modelCount) * float64(nextBucketsCount))

				// Place element in the correct bucket
				if nextBucketsSizes[pos] >= nextBucketCapacity {
					// if the bucket was full, put in in spill bucket
					*spillBucket = append(*spillBucket, element)
				} else {
					// write the element in the next empty space in its bucket
					writeArr[pos*nextBucketCapacity+nextBucketsSizes[pos]] = element
					nextBucketsSizes[pos]++
				}
			}
			offset += currentBucketCapacity
		}

		// check if breaking the loop
		tmpBucketCapacity := nextBucketCapacity / config.FanOut
		if tmpBucketCapacity < config.Threshold {
			return writeArr
		}

		currentBucketCapacity = nextBucketCapacity
		config.BucketCapacity = tmpBucketCapacity
		config.BucketsCount = config.InputSize / config.BucketCapacity

		// swap
		readArr, writeArr = writeArr, readArr
	}
}

func reorderBucketElements(readArr []float64, model RMIModel, config *LearnedSortingConfig, inferCache map[float64]float64) {
	// We use an auxiliary array to store cached pos for each bucket.
	// Because we use the exact calculations in 2 places.
	// Then we rewrite the cachedPos for the next bucket in the same array.
	cachedPos := make([]int, config.BucketCapacity)

	offset := 0
	for i := 0; i < config.BucketsCount; i++ {
		// to store counts of each element in a bucket
		countArr := make([]int, config.BucketsSizes[i])

		// record the counts of the predicted position
		for j := 0; j < config.BucketsSizes[i]; j++ {
			element := readArr[offset+j]

			// var pos int
			// if cachedInfer, found := inferCache[element]; found {
			// 	pos = int(cachedInfer * float64(config.InputSize))
			// } else {
			// 	// must calculate in [0, bucketsCount) range
			// 	inferResult := infer(model, element, config.ModelCount)
			// 	pos = int(inferResult * float64(config.InputSize))
			// 	inferCache[element] = inferResult
			// }
			cachedInfer := inferCache[element]
			pos := int(cachedInfer * float64(config.InputSize))
			// pos := int(infer(model, element, config.ModelCount) * float64(config.InputSize))

			// convert to local pos
			pos = pos - offset
			// in a few (hopefully) cases we might predict above the range, this fixes it
			pos = int(math.Min(float64(pos), float64(config.BucketsSizes[i]-1)))
			cachedPos[j] = pos

			countArr[pos]++
		}

		// Calculate the cumulative count (checkout Counting Sort to understand this step)
		for j := 1; j < len(countArr); j++ {
			countArr[j] += countArr[j-1]
		}

		// order elements using the cumulative counts in prev step
		reorderedBucket := make([]float64, config.BucketsSizes[i])
		for j := 0; j < config.BucketsSizes[i]; j++ {
			element := readArr[offset+j]
			// we've already calculated it
			pos := cachedPos[j]
			// insert the element in the calculated position
			// why -1? because countArray counts from 1,
			// while when we use it as index, we count from 0.
			reorderedBucket[countArr[pos]-1] = element
			countArr[pos]--
		}

		copy(readArr[offset:offset+config.BucketsSizes[i]], reorderedBucket)
		offset += config.BucketCapacity
	}
}

// removes unfilled parts of the slice
func compact(arr []float64, config *LearnedSortingConfig) []float64 {
	compactSlice := []float64{}
	for i, size := range config.BucketsSizes {
		// or you can do it by addition
		offset := i * config.BucketCapacity
		compactSlice = append(compactSlice, arr[offset:offset+size]...)
	}

	return compactSlice
}

func insertionSort(arr []float64) {
	for i := 1; i < len(arr); i++ {
		key := arr[i]
		j := i - 1

		// Find the location where the key should be inserted
		for j >= 0 && arr[j] > key {
			arr[j+1] = arr[j] // Shift elements to the right
			j--
		}
		arr[j+1] = key // Insert the key in the correct position
	}
}

// Merge function merges two sorted arrays into one sorted array efficiently.
func merge(arr1, arr2 []float64) []float64 {
	n, m := len(arr1), len(arr2)
	result := make([]float64, 0, n+m) // Pre-allocate the result slice with capacity of n + m
	i, j := 0, 0

	// Merge the arrays
	for i < n && j < m {
		if arr1[i] < arr2[j] {
			result = append(result, arr1[i])
			i++
		} else {
			result = append(result, arr2[j])
			j++
		}
	}

	// Append any remaining elements from arr1 or arr2
	result = append(result, arr1[i:]...)
	result = append(result, arr2[j:]...)

	return result
}

// infer function implements the logic of the original JavaScript function
func infer(model RMIModel, elementKey float64, modelCount []int) float64 {
	r := 0
	layerIndex := 0

	// up until the last layer, we want to trunc the predicted number
	// so that it becomes an integer, so we use it as index for the model in next layer
	for ; layerIndex < len(model)-1; layerIndex++ {
		// during "distribute training data to the next layer",
		// we scaled the output to to number of models in the next layer
		// so instead of predicting [0, 1), it predicts [0, n)
		value := elementKey*model[layerIndex][r].Slope + model[layerIndex][r].Intercept
		if value <= 0 {
			r = 0
		} else {
			r = int(math.Min(value, float64(modelCount[layerIndex+1])-0.0001))
		}
	}

	// In the last layer, predict the actual cdf value [0, 1)
	// NOTE: layerIndex is correctly at last element index
	value := elementKey*model[layerIndex][r].Slope + model[layerIndex][r].Intercept
	if value < 0 {
		return 0
	}
	// TODO replace with value >= 1?
	if value > 0.9999 {
		value = 0.9999
	}
	return value
}

// distDataNextLayer distributes data to the next layer in-place
func distDataNextLayer(currData []TrainingData, currModel LinearModel, nextLayerData [][]TrainingData, nextLayerModelCount int) {
	for _, element := range currData {
		// predict which model to pass data to
		i := int(math.Min(math.Max(element.Key*currModel.Slope+currModel.Intercept, 0), float64(nextLayerModelCount)-0.0001))
		nextLayerData[i] = append(nextLayerData[i], element)
	}
}

type TrainingData struct {
	Key float64
	CDF float64
}

// Train trains an RMI model using the given data, layers, and model counts
func Train(arr []float64, layersCount int, modelCount []int) RMIModel {
	// Sample 1% of data, but not above 100 or below 10
	samples := sample(arr, 0.01, 10)
	sort.Float64s(samples)

	// Initialize training sets [layer][model][data]
	trainingSets := make([][][]TrainingData, layersCount)
	for i := 0; i < layersCount; i++ {
		trainingSets[i] = make([][]TrainingData, modelCount[i])
	}

	// Insert data into the root model
	for i, sample := range samples {
		trainingSets[0][0] = append(trainingSets[0][0], TrainingData{
			Key: sample,
			CDF: float64(i) / float64(len(samples)-1),
		})
	}

	// Initialize the CDF model
	cdfModel := make([][]LinearModel, layersCount)
	for i := 0; i < layersCount; i++ {
		cdfModel[i] = make([]LinearModel, modelCount[i])
	}

	// if a model receives no training data, uses this model
	// make sure to deep copy when using
	defaultModel := simpleLinearModel(trainingSets[0][0], LinearModel{})

	// Train the models layer by layer
	for layerIndex := 0; layerIndex < layersCount; layerIndex++ {
		for modelIndex := 0; modelIndex < modelCount[layerIndex]; modelIndex++ {
			// Train the model with the associated training set
			cdfModel[layerIndex][modelIndex] = simpleLinearModel(trainingSets[layerIndex][modelIndex], defaultModel)

			// Distribute training data to the next layer
			if layerIndex+1 < layersCount {
				// Scale the slope and intercept for the next layer
				slope := cdfModel[layerIndex][modelIndex].Slope * float64(modelCount[layerIndex+1])
				intercept := cdfModel[layerIndex][modelIndex].Intercept * float64(modelCount[layerIndex+1])

				// Update the model
				cdfModel[layerIndex][modelIndex] = LinearModel{Slope: slope, Intercept: intercept}

				// Distribute the data to the next layer
				distDataNextLayer(trainingSets[layerIndex][modelIndex], cdfModel[layerIndex][modelIndex], trainingSets[layerIndex+1], modelCount[layerIndex+1])
			}
		}
	}

	return cdfModel
}

// simpleLinearModel calculates the slope and intercept for a linear model based on the training data
func simpleLinearModel(trainingData []TrainingData, defaultModel LinearModel) LinearModel {
	if len(trainingData) == 0 {
		return LinearModel{
			Slope:     defaultModel.Slope,
			Intercept: defaultModel.Intercept,
		}
	}

	// Initialize variables to track the min and max values for key and cdf
	minKey := math.Inf(1)  // Positive infinity
	maxKey := math.Inf(-1) // Negative infinity
	minTarget := math.Inf(1)
	maxTarget := math.Inf(-1)

	// Iterate through the training data to find the min and max of key and cdf
	for _, data := range trainingData {
		if data.Key < minKey {
			minKey = data.Key
		}
		if data.Key > maxKey {
			maxKey = data.Key
		}
		if data.CDF < minTarget {
			minTarget = data.CDF
		}
		if data.CDF > maxTarget {
			maxTarget = data.CDF
		}
	}

	// Calculate the slope and intercept of the linear model
	// this logic is to avoid division by a value too close to zero.
	deltaKey := maxKey - minKey
	if deltaKey < 1 {
		deltaKey = 1
	}
	slope := (maxTarget - minTarget) / deltaKey
	intercept := minTarget - slope*minKey

	// Return the computed slope and intercept
	return LinearModel{
		Slope:     slope,
		Intercept: intercept,
	}
}

// sampleWithReplacement performs uniform random sampling with replacement
func sampleWithReplacement(arr []float64, sampleSize int) []float64 {
	// Create a slice to hold the samples
	samples := make([]float64, 0, sampleSize)

	// Sample with replacement
	for len(samples) < sampleSize {
		index := rand.Intn(len(arr)) // Random index within the array
		samples = append(samples, arr[index])
	}

	return samples
}

// sample samples a percentage of data, with a minimum size
// ptg [0, 1] samples `ptg`% of data
// minSize unless array is smaller
func sample(arr []float64, ptg float64, minSize int) []float64 {
	// If the array is smaller than the minSize, return the array as is
	if len(arr) <= minSize {
		// deep copy
		newSlice := make([]float64, len(arr))
		copy(newSlice, arr)
		return newSlice
	}

	// sample {ptg}% of data, but not bigger than maxSize
	sampleSize := int(float64(len(arr)) * ptg)
	if sampleSize < minSize {
		sampleSize = minSize
	}

	// Return the sampled data using sampleWithReplacement
	return sampleWithReplacement(arr, sampleSize)
}
