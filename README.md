# LearnedSort

[![Go Reference](https://pkg.go.dev/badge/github.com/amirreza-hos/learnedsort.svg)](https://pkg.go.dev/github.com/amirreza-hos/learnedsort)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/amirreza-hos/learnedsort)](https://goreportcard.com/report/github.com/amirreza-hos/learnedsort)
<meta name="google-site-verification" content="iIlZK_ZQ_Khbe8nbOjGw-_dgs1E4xqZWtrhWEQ_NcFE" />

A Go module to **sort** your float64 arrays **faster** than the standard sort, using the LearnedSort algorithm. Checkout **[LearnedSort Algorithm](#learnedsort-algorithm)** section for more information.

---

## Installation

To install the module, run:

```bash
go get github.com/amirreza-hos/learnedsort
```

## Usage

```go
package main

import "github.com/amirreza-hos/learnedsort"

func main() {
    // model configuration
    modelCount := []int{1, 10}
    modelLayers := 2
    // learned sort configuration
	threshold := 10
	fanOut := 10000
    model := learnedsort.Train(arr, modelLayers, modelCount)
	learnedsort.LearnedSort(arr, model, fanOut, threshold, modelCount)
}
```

## LearnedSort Algorithm

This module is based on **[The Case for a Learned Sorting Algorithm](https://doi.org/10.1145/3318464.3389752)** paper. The idea is to train a model (RMI model) on a small portion of your array, then use the model to predict the sorted position of each element of your array, and finally do an insertion sort (which is very efficient for partially sorted arrays) to make sure the array is completely sorted. The paper also incorporates 
"bucketization" to make the algorithm cache friendly.

This module omits the "In-bucket reordering" part of the original algorithm, as this part seems to make the algorithm slower. My explanation is that when you do the first step, bucketization, with threshold=10, buckets are small enough to use insertion sort on them, and insertion sort is more efficient than the "In-bucket reordering" part of the algorithm.

## Benchmark

The code used to perform the benchmark:

```go
func generateRandomIntArr(length, min, max int) []float64 {
	arr := make([]float64, length)
	for i := 0; i < length; i++ {
		// Generate a random number between min and max (inclusive)
		arr[i] = float64(rand.Intn(max-min+1) + min)
	}
	return arr
}

arr := generateRandomIntArr(100000, 0, 100000)
modelCount := []int{1, 10}
threshold := 10
fanOut := 10000

startDefault := time.Now()
sort.Float64s(arrCopy1)
elapsedDefault := time.Since(startDefault)

startLearned := time.Now()
model := learnedsort.Train(arr, 2, modelCount)
learnedsort.LearnedSort(arrCopy2, model, fanOut, threshold, modelCount)
elapsedLearned := time.Since(startLearned)
```

Result on go1.21 for 100000 elements:

**LearnedSort** took **6.987043ms**, **default sort** took **16.214177ms**.
