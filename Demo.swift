// Run this script from a Terminal on macOS using the command:
// $ cat SEFR.swift Demo.swift | swift -

import Foundation
import Vision
import QuartzCore

// Because Vision gives us 2048 features per image, but we only have a few
// images, the classification problem isn't very hard. To make it harder,
// set useRandomFeatures to true. We will now pick a small number of features 
// at random and try to do the classification with those.
let useRandomFeatures = false
let numFeatures = 5
let indices = (0..<numFeatures).map { _ in Int.random(in: 0..<2048) }

func contentsOfDirectory(at url: URL) -> [URL] {
  guard let results = try? FileManager.default.contentsOfDirectory(at: url,
                includingPropertiesForKeys: [],  options: .skipsHiddenFiles) else {
    fatalError("Could not read folder: \(url)")
  }
  return results
}

let trainURL = URL(fileURLWithPath: "images/train")
let testURL = URL(fileURLWithPath: "images/test")

// The classes are the names of the subfolders inside the "train" folder.
let classes = contentsOfDirectory(at: trainURL).map { $0.lastPathComponent }.sorted()
print("Found classes:", classes.joined(separator: ", "))

var trainExamples: [[Float]] = []
var trainLabels: [Int] = []
var testExamples: [[Float]] = []
var testLabels: [Int] = []

let request = VNGenerateImageFeaturePrintRequest()
request.imageCropAndScaleOption = .scaleFill

// This uses Vision FeaturePrint.Scene to create a feature vector for an image.
func generateFeatures(url: URL) -> [Float] {
  var values: [Float] = []
  do {
    let handler = VNImageRequestHandler(url: url, options: [:])
    try handler.perform([request])

    if let observations = request.results as? [VNFeaturePrintObservation],
      let observation = observations.first, observation.elementType == .float {

      // Copy the contents of the VNFeaturePrintObservation into a Float array.
      observation.data.withUnsafeBytes { raw in
        let ptr = raw.baseAddress!.assumingMemoryBound(to: Float.self)
        values = Array(UnsafeBufferPointer(start: ptr, count: observation.elementCount))
        //print(values.min(), values.max())

        if useRandomFeatures {
          values = indices.map { values[$0] }
        }
      }
    }
  } catch {
    print("Error: Could not generate image features: \(error)")
  }
  return values
}

// Loop through the "train" and "test" folders...
for isTraining in [true, false] {
  print("Generating features for dataset:", isTraining ? "train" : "test")

  // For every class...
  for (classIndex, className) in classes.enumerated() {
    print("\tclass: \(className)", terminator: "")

    let classURL = (isTraining ? trainURL : testURL).appendingPathComponent(className)
    let imageFiles = contentsOfDirectory(at: classURL)

    // For every image file in this class...
    for imageURL in imageFiles {
      let features = generateFeatures(url: imageURL)
      if isTraining {
        trainExamples.append(features)
        trainLabels.append(classIndex)
      } else {
        testExamples.append(features)
        testLabels.append(classIndex)
      }
      print(".", terminator: "")
    }
    print("")
  }
}

print("Training examples: \(trainExamples.count), features: \(trainExamples[0].count)")
print("Testing examples: \(testExamples.count), features: \(testExamples[0].count)")

// It appears that VNFeaturePrintObservation always contains values >= 0.
// That's good because SEFR works best when all features are positive. 
// If your data contains negative values, consider normalizing the training 
// data so it's in the range [0, 1].

// Train the model. For SEFR the training set does not have to be shuffled.
let trainingStartTime = CACurrentMediaTime()
let model = SEFRMulticlass()
model.fit(examples: trainExamples, targets: trainLabels)
print("Training time: \(CACurrentMediaTime() - trainingStartTime) sec.")

// Run inference on the test set.
let predictionStartTime = CACurrentMediaTime()
let predictions = model.predict(examples: testExamples)
print("Prediction time: \(CACurrentMediaTime() - predictionStartTime) sec.")

// What is the test accuracy?
var correct = 0
for (trueLabel, predLabel) in zip(testLabels, predictions) {
  if trueLabel == predLabel {
    correct += 1
  }
}
print("Accuracy: \(100*Float(correct)/Float(testLabels.count))%")
print("True labels:")
print(testLabels)
print("Predicted labels:")
print(predictions)

// Run the model on an image for which we don't have a label.
let features = generateFeatures(url: URL(fileURLWithPath: "images/5265478313_8eb626c78a_o.jpg"))
let prediction = model.predict(example: features)
print("Cat+balloon image prediction:", classes[prediction])

// To inspect what the model has learned, you can look at the weights & bias:
//print(model.classifiers[0].weights)
//print(model.classifiers[0].bias)
