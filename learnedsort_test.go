package learnedsort

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sort"
	"testing"
	"time"
	"unsafe"
)

// TODO change len(arr) to the new readArr size, like original LearnedSort
// TODO pre-allocate spill bucket in tests, like original LearnedSort

func isCloseTo(a, b, tolerance float64) bool {
	return math.Abs(a-b) <= tolerance
}

// generateRandomIntArray generates a random slice of integers with the given length, min, and max values.
// note that the values are ints but stores as float64, don't ask why.
func generateRandomIntArr(length, min, max int) []float64 {
	arr := make([]float64, length)
	for i := 0; i < length; i++ {
		// Generate a random number between min and max (inclusive)
		arr[i] = float64(rand.Intn(max-min+1) + min)
	}
	return arr
}

// Utility function to check if an element exists in a slice
func contains[T comparable](slice []T, elem T) bool {
	for _, v := range slice {
		if v == elem {
			return true
		}
	}
	return false
}

func TestInfer(t *testing.T) {
	t.Run("infer correctly", func(t *testing.T) {
		cdfModel := RMIModel{
			// layer 0
			{
				// model 0
				{Slope: 0.7 * 3, Intercept: 0},
			},
			// layer 1
			{
				// model 0, 1, 2
				{Slope: 0, Intercept: 0},
				{Slope: 0, Intercept: 0},
				{Slope: 0.5, Intercept: 0},
			},
		}

		// Layer 0:
		// 0.7 [slope] * 1 [input] * 3 [models in next layer] + 0 [intercept] = 2.1
		// => model 2 in next layer
		// Layer 1 (last layer):
		// 0.5 [slope] * 1 [input] * 1 [last layer] + 0 [intercept] = 0.5

		result := infer(cdfModel, 1, []int{1, 3})
		expected := 0.5
		if math.Abs(result-expected) > 1e-9 {
			t.Errorf("Expected %v, but got %v", expected, result)
		}
	})

	t.Run("should not infer below 0, should not throw index error", func(t *testing.T) {
		cdfModel := RMIModel{
			// layer 0
			{
				// model 0
				{Slope: 0.1 * 3, Intercept: -2},
			},
			// layer 1
			{
				// model 0, 1, 2
				{Slope: 0.5, Intercept: 0},
				{Slope: 0, Intercept: 0},
				{Slope: 0, Intercept: 0},
			},
		}

		// Layer 0:
		// 0.1 [slope] * 1 [input] * 3 [models in next layer] + -2 [intercept] = -1.7
		// => model 0 in next layer
		// Layer 1 (last layer):
		// 0.5 [slope] * 1 [input] * 1 [last layer] + 0 [intercept] = 0.5

		result := infer(cdfModel, 1, []int{1, 3})
		expected := 0.5
		if math.Abs(result-expected) > 1e-9 {
			t.Errorf("Expected %v, but got %v", expected, result)
		}
	})

	t.Run("should infer correctly with 3 layers", func(t *testing.T) {
		cdfModel := RMIModel{
			// layer 0
			{
				// model 0
				{Slope: 0.1 * 3, Intercept: -2},
			},
			// layer 1
			{
				// model 0, 1, 2
				{Slope: 0.4 * 6, Intercept: 1},
				{Slope: 0, Intercept: 0},
				{Slope: 0, Intercept: 0},
			},
			// layer 2
			{
				// model 0, 1, 2, 3, 4, 5
				{Slope: 0, Intercept: 0},
				{Slope: 0, Intercept: 0},
				{Slope: 0, Intercept: 0},
				{Slope: 0.2, Intercept: 0},
				{Slope: 0, Intercept: 0},
				{Slope: 0, Intercept: 0},
			},
		}

		// Layer 0:
		// 0.1 [slope] * 1 [input] * 3 [models in next layer] + -2 [intercept] = -1.7
		// => model 0 in next layer
		// Layer 1:
		// 0.4 [slope] * 1 [input] * 6 [models in next layer] + 1 [intercept] = 3.4
		// => model 3 in next layer
		// Layer 2:
		// 0.2 [slope] * 1 [input] * 1 [last layer] + 0 [intercept] = 0.2

		result := infer(cdfModel, 1, []int{1, 3, 6})
		expected := 0.2
		if math.Abs(result-expected) > 1e-9 {
			t.Errorf("Expected %v, but got %v", expected, result)
		}
	})
}

func TestDistDataNextLayer(t *testing.T) {
	t.Run("should distribute correctly and modify the original array", func(t *testing.T) {
		data := []TrainingData{
			{Key: 10, CDF: 0},
			{Key: 50, CDF: 0.5},
			{Key: 100, CDF: 1},
		}
		nextLayerModelCount := 3
		model := LinearModel{
			Slope:     0.01107 * float64(nextLayerModelCount),
			Intercept: -0.0904 * float64(nextLayerModelCount),
		}
		nextLayerData := make([][]TrainingData, nextLayerModelCount)

		distDataNextLayer(data, model, nextLayerData, nextLayerModelCount)

		expected := [][]TrainingData{
			{data[0]}, // Model 0
			{data[1]}, // Model 1
			{data[2]}, // Model 2
		}

		// we expect it to modify the original nextLayerData
		// as it's passing slice, we don't need explicit pass by reference

		if !reflect.DeepEqual(nextLayerData, expected) {
			t.Errorf("Expected %v, but got %v", expected, nextLayerData)
		}
	})
}

func TestSampleWithReplacement(t *testing.T) {
	arr := []float64{10, 20, 30, 40, 50}
	result := sampleWithReplacement(arr, 10)

	if len(result) != 10 {
		t.Errorf("Expected %v, but got %v", 10, len(result))
	}
}

func TestSample(t *testing.T) {
	t.Run("should return the array itself (copy), as it's too small to sample", func(t *testing.T) {
		arr := []float64{10, 20, 30, 40, 50}
		result := sample(arr, 0.01, 10)

		if !reflect.DeepEqual(arr, result) {
			t.Errorf("Expected %v, but got %v", arr, result)
		}
	})
	t.Run("should return a copy of the array, not the original", func(t *testing.T) {
		arr := []float64{10, 20, 30, 40, 50}
		result := sample(arr, 0.01, 10)

		ptr1 := unsafe.SliceData(arr)
		ptr2 := unsafe.SliceData(result)

		if ptr1 == ptr2 {
			t.Errorf("sample returned the same array, while we expected a copy")
		}
	})
}

func TestSimpleLinearModel(t *testing.T) {
	t.Run("should predict the exact value for min and max", func(t *testing.T) {
		data := []TrainingData{
			{Key: 2000, CDF: 0},
			{Key: 6000, CDF: 0.5},
			{Key: 12000, CDF: 1},
		}
		output := simpleLinearModel(data, LinearModel{})
		pred0 := output.Slope*data[0].Key + output.Intercept
		pred1 := output.Slope*data[len(data)-1].Key + output.Intercept
		if !isCloseTo(pred0, data[0].CDF, 0.01) {
			t.Errorf("Expected prediction for min value close to %v, but got %v", data[0].CDF, pred0)
		}
		if !isCloseTo(pred1, data[len(data)-1].CDF, 0.01) {
			t.Errorf("Expected prediction for max value close to %v, but got %v", data[len(data)-1].CDF, pred1)
		}
	})

	t.Run("should predict the exact value for min and max (different data)", func(t *testing.T) {
		data := []TrainingData{
			{Key: 2000, CDF: 0.1},
			{Key: 6000, CDF: 0.5},
			{Key: 12000, CDF: 0.9},
		}
		output := simpleLinearModel(data, LinearModel{})
		pred0 := output.Slope*data[0].Key + output.Intercept
		pred1 := output.Slope*data[len(data)-1].Key + output.Intercept
		if !isCloseTo(pred0, data[0].CDF, 0.01) {
			t.Errorf("Expected prediction for min value close to %v, but got %v", data[0].CDF, pred0)
		}
		if !isCloseTo(pred1, data[len(data)-1].CDF, 0.01) {
			t.Errorf("Expected prediction for max value close to %v, but got %v", data[len(data)-1].CDF, pred1)
		}
	})

	t.Run("for input close to upper bound, should not predict below 0.5", func(t *testing.T) {
		data := []TrainingData{
			{Key: 2000, CDF: 0.1},
			{Key: 4000, CDF: 0.25},
			{Key: 10000, CDF: 0.75},
			{Key: 12000, CDF: 0.9},
		}
		output := simpleLinearModel(data, LinearModel{})
		pred0 := output.Slope*data[0].Key*1.01 + output.Intercept
		pred1 := output.Slope*data[len(data)-1].Key*0.99 + output.Intercept
		if pred0 > 0.5 {
			t.Errorf("Expected prediction for first value to be <= 0.5, but got %v", pred0)
		}
		if pred1 < 0.5 {
			t.Errorf("Expected prediction for last value to be >= 0.5, but got %v", pred1)
		}
	})

	t.Run("after scaling by n, should predict in range [0, n]", func(t *testing.T) {
		data := []TrainingData{
			{Key: 2000, CDF: 0.1},
			{Key: 6000, CDF: 0.5},
			{Key: 12000, CDF: 0.9},
		}
		output := simpleLinearModel(data, LinearModel{})

		// scaling
		output.Slope *= 3
		output.Intercept *= 3

		pred0 := output.Slope*data[0].Key*1.01 + output.Intercept
		pred1 := output.Slope*data[len(data)-1].Key*0.99 + output.Intercept

		if pred0 < 0 || pred0 > 1 {
			t.Errorf("Expected scaled prediction for first value in range [0, 1], but got %v", pred0)
		}
		if pred1 < 2 || pred1 > 3 {
			t.Errorf("Expected scaled prediction for last value in range [2, 3], but got %v", pred1)
		}
	})
}

func TestTrain(t *testing.T) {

	// Test case: train small array
	t.Run("TrainSmallArray", func(t *testing.T) {
		data := []float64{100, 10, 1, 2, 3, 1000}
		modelCount := []int{1, 100}

		cdfModel := Train(data, 2, modelCount)

		if !reflect.DeepEqual(data, []float64{100, 10, 1, 2, 3, 1000}) {
			t.Errorf("train() has modified the original array")
		}

		if cdfModel == nil {
			t.Errorf("train() returned nil, expected a valid 2D slice")
		}
		if len(cdfModel) == 0 || len(cdfModel[0]) == 0 {
			t.Errorf("train() returned an empty model, expected at least one LinearModel")
		}
		if reflect.TypeOf(cdfModel[0][0]) != reflect.TypeOf(LinearModel{}) {
			t.Errorf("train() returned invalid model type, expected LinearModel, got %v", reflect.TypeOf(cdfModel[0][0]))
		}
		if reflect.TypeOf(cdfModel[1][99]) != reflect.TypeOf(LinearModel{}) {
			t.Errorf("train() returned invalid model type at cdfModel[1][99], expected LinearModel, got %v", reflect.TypeOf(cdfModel[1][99]))
		}
	})

	// Test case: train small array on 4 layers
	t.Run("TrainSmallArrayFourLayers", func(t *testing.T) {
		data := []float64{100, 10, 1, 2, 3, 1000}
		modelCount := []int{1, 3, 6, 12}
		cdfModel := Train(data, 4, modelCount)

		if cdfModel == nil {
			t.Errorf("train() returned nil, expected a valid 2D slice")
		}
		if len(cdfModel) != 4 {
			t.Errorf("train() expected 4 layers, got %d", len(cdfModel))
		}
		if reflect.TypeOf(cdfModel[0][0]) != reflect.TypeOf(LinearModel{}) {
			t.Errorf("train() returned invalid model type, expected LinearModel, got %v", reflect.TypeOf(cdfModel[0][0]))
		}
		if reflect.TypeOf(cdfModel[1][2]) != reflect.TypeOf(LinearModel{}) {
			t.Errorf("train() returned invalid model type at cdfModel[1][2], expected LinearModel, got %v", reflect.TypeOf(cdfModel[1][2]))
		}
		if reflect.TypeOf(cdfModel[2][5]) != reflect.TypeOf(LinearModel{}) {
			t.Errorf("train() returned invalid model type at cdfModel[2][5], expected LinearModel, got %v", reflect.TypeOf(cdfModel[2][5]))
		}
		if reflect.TypeOf(cdfModel[3][11]) != reflect.TypeOf(LinearModel{}) {
			t.Errorf("train() returned invalid model type at cdfModel[3][11], expected LinearModel, got %v", reflect.TypeOf(cdfModel[3][11]))
		}
	})
}

func TestTrainInfer(t *testing.T) {
	t.Run("TrainLargeArrayPredictions", func(t *testing.T) {
		modelCount := []int{1, 100}

		// Generate a random array with 1000 integers between 0 and 1000
		arr := generateRandomIntArr(1000, 0, 1000)

		copyArr := make([]float64, len(arr))
		copy(copyArr, arr)
		sort.Float64s(copyArr)
		smallest := copyArr[0]
		biggest := copyArr[len(copyArr)-1]

		// Train the model
		cdfModel := Train(arr, 2, modelCount)

		// Infer the values for the smallest and biggest
		smallestCdf := infer(cdfModel, smallest, modelCount)
		biggestCdf := infer(cdfModel, biggest, modelCount)

		// Assertions
		if smallestCdf >= 0.1 {
			t.Errorf("Expected smallestCdf < 0.1, got %v", smallestCdf)
		}
		if biggestCdf <= 0.9 {
			t.Errorf("Expected biggestCdf > 0.9, got %v", biggestCdf)
		}
	})
}

// TODO test second recursion
func TestBucketize(t *testing.T) {
	t.Run("bucketize once", func(t *testing.T) {
		// Test data as per the JavaScript example
		readArr := []float64{9, 3, 2, 0, 7, 0, 7, 8, 5, 0}
		spillBucket := []float64{}
		config := LearnedSortingConfig{
			ModelCount:     []int{1, 3},
			BucketsCount:   5,
			BucketCapacity: 2,
			BucketsSizes:   nil,
			Threshold:      2,
			InputSize:      len(readArr),
			FanOut:         5,
		}
		config.BucketsSizes = make([]int, config.BucketsCount)
		for i := range config.BucketsSizes {
			config.BucketsSizes[i] = config.BucketCapacity
		}

		spillBucket = append(spillBucket, readArr[config.BucketsCount*config.BucketCapacity:]...)
		readArr = readArr[:config.BucketsCount*config.BucketCapacity]
		writeArr := make([]float64, config.BucketsCount*config.BucketCapacity)

		model := Train(readArr, 2, config.ModelCount)
		inferCache := make(map[float64]float64)

		readArr = bucketize(readArr, writeArr, &spillBucket, model, &config, inferCache)

		// given bucketsCount:5 bucketCapacity:2 threshold:2 inputSize:10
		// recursion must happen once, config must stay the same,
		if config.BucketCapacity != 2 {
			t.Errorf("Expected bucketCapacity to be 2, got %d", config.BucketCapacity)
		}
		if config.BucketsCount != 5 {
			t.Errorf("Expected BucketsCount to be 5, got %d", config.BucketsCount)
		}

		// updated readArr must have max of 10 elements,
		expectedLength := config.BucketsCount * config.BucketCapacity
		if len(readArr) != expectedLength {
			t.Errorf("Expected readArr length to be %d, got %d", expectedLength, len(readArr))
		}

		// three 0s can't fit in a single bucket, so spill bucket must include 0
		if !contains(spillBucket, 0) {
			t.Errorf("Expected spillBucket to contain 0")
		}

		// Verify first and last elements in readArr
		if readArr[0] != 0 {
			t.Errorf("Expected readArr[0] to be 0, got %f", readArr[0])
		}
		if readArr[len(readArr)-1] != 9 && readArr[len(readArr)-2] != 9 {
			t.Errorf("Expected the last or second to last element to be 9")
		}
	})

	t.Run("produce unfilled bucket, should not interfere with overall process", func(t *testing.T) {
		readArr := []float64{2, 1, 0, 14, 14, 14, 14, 14}

		readArrCopy := make([]float64, len(readArr))
		copy(readArrCopy, readArr)

		spillBucket := []float64{}
		config := LearnedSortingConfig{
			ModelCount:     []int{1, 3},
			BucketsCount:   2,
			BucketCapacity: 4,
			BucketsSizes:   nil,
			Threshold:      2,
			InputSize:      len(readArr),
			FanOut:         2,
		}
		config.BucketsSizes = make([]int, config.BucketsCount)
		for i := range config.BucketsSizes {
			config.BucketsSizes[i] = config.BucketCapacity
		}

		spillBucket = append(spillBucket, readArr[config.BucketsCount*config.BucketCapacity:]...)
		readArr = readArr[:config.BucketsCount*config.BucketCapacity]
		writeArr := make([]float64, config.BucketsCount*config.BucketCapacity)

		model := Train(readArr, 2, config.ModelCount)
		inferCache := make(map[float64]float64)

		readArr = bucketize(readArr, writeArr, &spillBucket, model, &config, inferCache)

		/*
			In the process of bucketize, our original elements go in readArr and spillBucket.
			We want to make sure elements in the modified readArr and spillBucket,
			match the elements in the original readArr.

			As we're using bucketization, we might not completely fill a bucket,
			and the unfilled part will either hold the previous values (as we're reusing arrays),
			or they might hold default value (0), so we use below approach to test we've done the correct job.
		*/
		sort.Float64s(readArrCopy)

		actualElements := []float64{}
		for i, size := range config.BucketsSizes {
			offset := i * config.BucketCapacity
			actualElements = append(actualElements, readArr[offset:offset+size]...)
		}
		allElements := append(actualElements, spillBucket...)
		sort.Float64s(allElements)

		if !reflect.DeepEqual(allElements, readArrCopy) {
			t.Errorf("bucketize produced %d extra/less elements.", len(allElements)-len(readArrCopy))
		}
	})

	t.Run("bucketize odd array, readArr + spill must have same elements as original", func(t *testing.T) {
		readArr := []float64{9, 2, 0, 7, 0, 7, 8, 5, 0}

		readArrCopy := make([]float64, len(readArr))
		copy(readArrCopy, readArr)

		spillBucket := []float64{}
		config := LearnedSortingConfig{
			ModelCount:     []int{1, 3},
			BucketsCount:   2,
			BucketCapacity: 4,
			BucketsSizes:   nil,
			Threshold:      4,
			InputSize:      len(readArr),
			FanOut:         2,
		}
		config.BucketsSizes = make([]int, config.BucketsCount)
		for i := range config.BucketsSizes {
			config.BucketsSizes[i] = config.BucketCapacity
		}

		spillBucket = append(spillBucket, readArr[config.BucketsCount*config.BucketCapacity:]...)
		readArr = readArr[:config.BucketsCount*config.BucketCapacity]
		writeArr := make([]float64, config.BucketsCount*config.BucketCapacity)

		model := Train(readArr, 2, config.ModelCount)
		inferCache := make(map[float64]float64)

		readArr = bucketize(readArr, writeArr, &spillBucket, model, &config, inferCache)

		allElements := append(readArr, spillBucket...)
		sort.Float64s(allElements)
		sort.Float64s(readArrCopy)
		if !reflect.DeepEqual(allElements, readArrCopy) {
			t.Errorf("bucketize produced %d extra/less elements.", len(allElements)-len(readArrCopy))
		}
	})

	t.Run("bucketize odd array twice, readArr + spill must have same elements as original", func(t *testing.T) {
		readArr := []float64{9, 2, 0, 7, 0, 7, 8, 5, 0}

		readArrCopy := make([]float64, len(readArr))
		copy(readArrCopy, readArr)

		spillBucket := []float64{}
		config := LearnedSortingConfig{
			ModelCount:     []int{1, 3},
			BucketsCount:   2,
			BucketCapacity: 4,
			BucketsSizes:   nil,
			Threshold:      2,
			InputSize:      len(readArr),
			FanOut:         2,
		}

		config.BucketsSizes = make([]int, config.BucketsCount)
		for i := range config.BucketsSizes {
			config.BucketsSizes[i] = config.BucketCapacity
		}

		spillBucket = append(spillBucket, readArr[config.BucketsCount*config.BucketCapacity:]...)
		readArr = readArr[:config.BucketsCount*config.BucketCapacity]
		writeArr := make([]float64, config.BucketsCount*config.BucketCapacity)

		model := Train(readArr, 2, config.ModelCount)
		inferCache := make(map[float64]float64)

		readArr = bucketize(readArr, writeArr, &spillBucket, model, &config, inferCache)

		allElements := append(readArr, spillBucket...)
		sort.Float64s(allElements)
		sort.Float64s(readArrCopy)
		if !reflect.DeepEqual(allElements, readArrCopy) {
			t.Errorf("bucketize produced %d extra/less elements.", len(allElements)-len(readArrCopy))
		}
	})
}

func TestReorderBucketElements(t *testing.T) {
	t.Run("reorder 2 buckets", func(t *testing.T) {
		readArr := []float64{0, 4, 2, 0, 0, 9, 7, 8, 5, 7}
		config := LearnedSortingConfig{
			ModelCount:     []int{1, 3},
			BucketsCount:   2,
			BucketCapacity: 5,
			BucketsSizes:   []int{5, 5},
			Threshold:      5,
			InputSize:      len(readArr),
			FanOut:         2,
		}

		model := Train(readArr, 2, config.ModelCount)

		inferCache := make(map[float64]float64)
		// as reorderBucketElements expects a filled cache, we must calculate it.
		for _, element := range readArr {
			if _, found := inferCache[element]; !found {
				inferCache[element] = infer(model, element, config.ModelCount)
			}
		}

		reorderBucketElements(readArr, model, &config, inferCache)

		// TODO test and make sure all duplicates exist

		for i := 0; i < len(readArr)-1; i++ {
			if readArr[i] > readArr[i+1] {
				t.Errorf("array not sorted at index %d: %f > %f", i, readArr[i], readArr[i+1])
			}
		}
	})

	t.Run("handling unfilled bucket", func(t *testing.T) {
		// the -1 must stay in place
		readArr := []float64{0, 4, 2, 0, -1, 9, 7, 8, 5, 7}

		readArrCopy := make([]float64, len(readArr))
		copy(readArrCopy, readArr)

		config := LearnedSortingConfig{
			ModelCount:     []int{1, 3},
			BucketsCount:   2,
			BucketCapacity: 5,
			BucketsSizes:   []int{4, 5},
			Threshold:      5,
			InputSize:      len(readArr),
			FanOut:         2,
		}

		model := Train(readArr, 2, config.ModelCount)

		inferCache := make(map[float64]float64)
		// as reorderBucketElements expects a filled cache, we must calculate it.
		for _, element := range readArr {
			if _, found := inferCache[element]; !found {
				inferCache[element] = infer(model, element, config.ModelCount)
			}
		}

		reorderBucketElements(readArr, model, &config, inferCache)

		// the -1 must stay in place
		if readArr[4] != -1 {
			t.Error("reorderBucketElements has touched unexpected range of bucket")
		}

		// contents of each bucket must exist in the end
		sort.Float64s(readArr[0:5])
		sort.Float64s(readArrCopy[0:5])
		if !reflect.DeepEqual(readArr[0:5], readArrCopy[0:5]) {
			t.Error("first bucket has different elements", readArr[0:5], readArrCopy[0:5])
		}
		sort.Float64s(readArr[5:10])
		sort.Float64s(readArrCopy[5:10])
		if !reflect.DeepEqual(readArr[5:10], readArrCopy[5:10]) {
			t.Error("second bucket has different elements", readArr[5:10], readArrCopy[5:10])
		}
	})
}

func TestBucketizeReorder(t *testing.T) {
	t.Run("having only a few misplacement", func(t *testing.T) {
		readArr := generateRandomIntArr(100000, 0, 100000)
		modelCount := []int{1, 10}
		threshold := 10
		fanOut := 10000
		inputSize := len(readArr)

		spillBucket := []float64{}
		config := LearnedSortingConfig{
			ModelCount:     modelCount,
			BucketsCount:   fanOut,
			BucketCapacity: inputSize / fanOut,
			BucketsSizes:   nil,
			Threshold:      threshold,
			InputSize:      inputSize,
			FanOut:         fanOut,
		}

		config.BucketsSizes = make([]int, config.BucketsCount)
		for i := range config.BucketsSizes {
			config.BucketsSizes[i] = config.BucketCapacity
		}

		spillBucket = append(spillBucket, readArr[config.BucketsCount*config.BucketCapacity:]...)
		readArr = readArr[:config.BucketsCount*config.BucketCapacity]
		config.InputSize = len(readArr)
		writeArr := make([]float64, config.BucketsCount*config.BucketCapacity)

		model := Train(readArr, 2, config.ModelCount)
		inferCache := make(map[float64]float64)

		readArr = bucketize(readArr, writeArr, &spillBucket, model, &config, inferCache)

		copy1 := make([]float64, len(readArr))
		copy(copy1, readArr)
		copy1 = compact(copy1, &config)
		copy11 := make([]float64, len(copy1))
		copy(copy11, copy1)
		sort.Float64s(copy11)
		misplaceCount := 0
		for i, _ := range copy1 {
			if copy1[i] != copy11[i] {
				misplaceCount++
			}
		}
		fmt.Println("BUCKETIZE", "totalCount", len(copy1), "misplaceCount", misplaceCount, "rate", float64(misplaceCount)/float64(len(copy1)))

		// reorderBucketElements(readArr, model, &config, inferCache)

		// copy2 := make([]float64, len(readArr))
		// copy(copy2, readArr)
		// copy2 = compact(copy2, &config)
		// copy22 := make([]float64, len(copy2))
		// copy(copy22, copy1)
		// sort.Float64s(copy22)
		// misplaceCount = 0
		// for i, _ := range copy1 {
		// 	if copy2[i] != copy22[i] {
		// 		misplaceCount++
		// 	}
		// }
		// fmt.Println("REORDER", "totalCount", len(copy2), "misplaceCount", misplaceCount, "rate", float64(misplaceCount)/float64(len(copy2)))
	})
}

func TestInsertionSort(t *testing.T) {
	t.Run("sort", func(t *testing.T) {
		arr := []float64{10, 1, 1500, 0.5, 50.5, 2, 3}
		insertionSort(arr)

		if !sort.Float64sAreSorted(arr) {
			t.Errorf("Array is not sorted: %v", arr)
		}
	})
}

func TestMerge(t *testing.T) {
	t.Run("merge", func(t *testing.T) {
		arr1 := []float64{1, 5, 10, 20}
		arr2 := []float64{6, 7, 8, 9, 11}
		output := merge(arr1, arr2)

		expected := append(arr1, arr2...)
		sort.Float64s(expected)

		if !reflect.DeepEqual(output, expected) {
			t.Errorf("Merged array is incorrect: got %v, expected %v", output, expected)
		}
	})
}

func TestCompact(t *testing.T) {
	t.Run("compact", func(t *testing.T) {
		readArr := []float64{0, 4, 2, 0, -1, 9, 7, 8, 5, 7}
		config := LearnedSortingConfig{
			ModelCount:     []int{1, 3},
			BucketsCount:   5,
			BucketCapacity: 2,
			BucketsSizes:   []int{1, 1, 1, 1, 1},
			// doesn't matter
			Threshold: 0,
			InputSize: 0,
			FanOut:    0,
		}

		output := compact(readArr, &config)
		expected := []float64{0, 2, -1, 7, 5}

		if !reflect.DeepEqual(output, expected) {
			t.Error("Don't match", output, expected)
		}
	})
}

func TestLearnedSort(t *testing.T) {
	t.Run("even length", func(t *testing.T) {
		arr := []float64{9, 3, 2, 0, 7, 0, 7, 8, 5, 0}
		modelCount := []int{1, 3}
		model := Train(arr, 2, modelCount)
		threshold := 2
		fanOut := 5

		output := LearnedSort(arr, model, fanOut, threshold, modelCount)
		expected := []float64{0, 0, 0, 2, 3, 5, 7, 7, 8, 9}

		if !reflect.DeepEqual(output, expected) {
			t.Errorf("Expected %v, got %v", expected, output)
		}
	})

	t.Run("odd length", func(t *testing.T) {
		arr := []float64{9, 3, 2, 0, 7, 11, 0, 7, 8, 5, 0}
		modelCount := []int{1, 3}
		model := Train(arr, 2, modelCount)
		threshold := 2
		fanOut := 5

		output := LearnedSort(arr, model, fanOut, threshold, modelCount)
		expected := []float64{0, 0, 0, 2, 3, 5, 7, 7, 8, 9, 11}

		if !reflect.DeepEqual(output, expected) {
			t.Errorf("Expected %v, got %v", expected, output)
		}
	})

	t.Run("negative numbers", func(t *testing.T) {
		arr := []float64{-9, -3, 1, 0, -7, -11, 0, -7, -8, -5, 0}
		modelCount := []int{1, 3}
		model := Train(arr, 2, modelCount)
		threshold := 2
		fanOut := 5

		output := LearnedSort(arr, model, fanOut, threshold, modelCount)
		expected := []float64{-11, -9, -8, -7, -7, -5, -3, 0, 0, 0, 1}

		if !reflect.DeepEqual(output, expected) {
			t.Errorf("Expected %v, got %v", expected, output)
		}
	})

	t.Run("float numbers", func(t *testing.T) {
		arr := []float64{1.9, 1.3, 1, 1.0, 1.7, 1.11, 1.0, 1.7, 1.8, 1.5, 1.0}
		modelCount := []int{1, 3}
		model := Train(arr, 2, modelCount)
		threshold := 2
		fanOut := 5

		output := LearnedSort(arr, model, fanOut, threshold, modelCount)
		expected := []float64{1, 1, 1, 1, 1.11, 1.3, 1.5, 1.7, 1.7, 1.8, 1.9}

		if !reflect.DeepEqual(output, expected) {
			t.Errorf("Expected %v, got %v", expected, output)
		}
	})

	t.Run("long array", func(t *testing.T) {
		arr := generateRandomIntArr(100000, 0, 10000)
		modelCount := []int{1, 1000}
		model := Train(arr, 2, modelCount)
		threshold := 100
		fanOut := 1000

		output := LearnedSort(arr, model, fanOut, threshold, modelCount)

		// TODO also make sure input and output have same elements

		if !sort.Float64sAreSorted(output) {
			t.Errorf("Array is not sorted: %v", output)
		}
	})

	t.Run("long array preserving all duplicates", func(t *testing.T) {
		arr := generateRandomIntArr(100000, 0, 10000)

		arrCopy := make([]float64, len(arr))
		copy(arrCopy, arr)

		modelCount := []int{1, 1000}
		model := Train(arr, 2, modelCount)
		threshold := 100
		fanOut := 1000

		output := LearnedSort(arr, model, fanOut, threshold, modelCount)

		sort.Float64s(arrCopy)

		if !reflect.DeepEqual(arrCopy, output) {
			t.Error("LearnedSort lost some values")
		}
	})

	t.Run("being faster than default sort", func(t *testing.T) {
		arr := generateRandomIntArr(100000, 0, 100000)
		modelCount := []int{1, 10}
		threshold := 10
		fanOut := 10000

		arrCopy1 := make([]float64, len(arr))
		copy(arrCopy1, arr)

		startDefault := time.Now()
		sort.Float64s(arrCopy1)
		elapsedDefault := time.Since(startDefault)

		arrCopy2 := make([]float64, len(arr))
		copy(arrCopy2, arr)

		startLearned := time.Now()
		model := Train(arr, 2, modelCount)
		LearnedSort(arrCopy2, model, fanOut, threshold, modelCount)
		elapsedLearned := time.Since(startLearned)

		fmt.Printf("LearnedSort took %v, default sort took %v\n", elapsedLearned, elapsedDefault)
		// t.Logf("LearnedSort took %v, default sort took %v", elapsedLearned, elapsedDefault)
	})
}
