import Foundation

/**
  The SEFR binary classifier from https://arxiv.org/abs/2006.04620

  Based on https://github.com/sefr-classifier/sefr/
  and https://github.com/alankrantas/sefr_multiclass_classifier/

  To use this on a multiclass task, create a separate SEFR instance for each
  class and pass the class label into `positiveLabel` when fitting. Then make
  the predictions using each classifier and choose the class that gives the 
  largest score.
*/
public class SEFR {
  private(set) public var weights: [Float] = []
  private(set) public var bias: Float = 0

  /**
    Trains the classifier.

    - parameters:
      - examples: the training examples. Array of shape (N, M) where N is the 
        number of examples and M is the number of features. For best results,
        pre-process the data so it contains no negative numbers.
      - targets: the training labels. Array of length N.
      - positiveLabel: the label that represents the positive class. All other
        labels will be considered the negative class.
  */
  public func fit(examples: [[Float]], targets: [Int], positiveLabel: Int = 1) {
    assert(examples.count == targets.count)

    let numExamples = examples.count
    if numExamples == 0 { return }

    let numFeatures = examples[0].count
    if numFeatures == 0 { return }

    // Count how many positive examples and how many negative examples.
    // We store these counts as Floats to avoid having to cast them later.
    var countPos: Float = 0
    var countNeg: Float = 0
    for i in 0..<numExamples {
      if targets[i] == positiveLabel {
        countPos += 1
      } else {
        countNeg += 1
      }
    }

    weights = []
    for f in 0..<numFeatures {
      // Compute the average value of each individual feature for all the
      // positive examples. Likewise for the negative examples.
      var sumPos: Float = 0
      var sumNeg: Float = 0
      for i in 0..<numExamples {
        if targets[i] == positiveLabel {
          sumPos += examples[i][f]
        } else {
          sumNeg += examples[i][f]
        }
      }

      let avgPos = sumPos / countPos
      let avgNeg = sumNeg / countNeg

      // Compute a weight for each feature. These weights are values between
      // -1 and +1. A feature with a weight close to +1 is highly correlated
      // with predicting the positive class. A feature with a weight close to
      // -1 is likely to predict the negative class. A weight close to 0 means
      // this feature is irrelevant. The epsilon prevents division by zero.
      let weight = (avgPos - avgNeg) / (avgPos + avgNeg + 0.0000001);
      weights.append(weight)
    }

    var sumPosScore: Float = 0
    var sumNegScore: Float = 0
    for i in 0..<numExamples {
      // For each example, multiply its feature values with the corresponding 
      // weights. This gives a single numerical score for each example. 
      // A higher score implies this example is likely positive; a lower score 
      // implies the example likely belongs to the negative class. (At this
      // point we don't know yet where the actual decision boundary lies.)
      var score: Float = 0
      for f in 0..<numFeatures {
        score += examples[i][f] * weights[f]
      }

      if targets[i] == positiveLabel {
        sumPosScore += score
      } else {
        sumNegScore += score
      }
    }

    // Compute the average values of the scores from the positive examples.
    // Likewise for the negative examples.
    let avgPosScore = sumPosScore / countPos
    let avgNegScore = sumNegScore / countNeg

    // Compute a bias, so that when we do `W*x + b` the example is positive if 
    // the score is > 0 and negative if the score < 0. The bias is calculated 
    // using a weighted average in case the number of positive examples in the 
    // training dataset is different from the number of negative examples.
    bias = -(countNeg * avgPosScore + countPos * avgNegScore) / (countNeg + countPos)
  }

  /**
    Makes a prediction for a single example. This returns the raw score.
  */
  public func predictScore(example: [Float]) -> Float {
    assert(weights.count == example.count, "model is not trained yet")
    var score = bias
    for f in 0..<weights.count {
      score += example[f] * weights[f]
    }    
    return score
  }

  /**
    Makes a prediction for a single example. Returns true for the positive
    class, false for the negative class.
  */
  public func predict(example: [Float]) -> Bool {
    predictScore(example: example) >= 0
  }

  /**
    Makes a prediction for a list of examples. This returns the raw scores.
  */
  public func predictScore(examples: [[Float]]) -> [Float] {
    examples.map(predictScore)
  }

  /**
    Makes a prediction for a list of examples. Returns true for the positive
    class, false for the negative class.
  */
  public func predict(examples: [[Float]]) -> [Bool] {
    examples.map(predict)
  }
}

/**
  Wrapper around SEFR that lets you use it for multiclass classification.
*/
public class SEFRMulticlass {
  private(set) public var labels: [Int] = []
  private(set) public var classifiers: [SEFR] = []

  public func fit(examples: [[Float]], targets: [Int]) {
    classifiers = []
    labels = Set(targets).sorted()
    for label in labels {
      let sefr = SEFR()
      sefr.fit(examples: examples, targets: targets, positiveLabel: label)
      classifiers.append(sefr)
    }
  }

  /**
    Makes a prediction for a single example. Returns the class label.
  */
  public func predict(example: [Float]) -> Int {
    assert(classifiers.count > 0, "model is not trained yet")
    var largest = -Float.greatestFiniteMagnitude
    var bestClass = 0
    for (label, clf) in zip(labels, classifiers) {
      let pred = clf.predictScore(example: example)
      if pred > largest {
        largest = pred
        bestClass = label
      }
    }
    return bestClass
  }

  /**
    Makes a prediction for a list of examples. Returns the class labels.
  */
  public func predict(examples: [[Float]]) -> [Int] {
    examples.map(predict)
  }
}
