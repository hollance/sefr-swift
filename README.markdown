# SEFR in Swift

This is the source code that accompanies my blog post [The SEFR classifier](https://machinethink.net/blog/sefr-classifier-in-swift/).

SEFR is a binary classifier. To use it on a binary classfication task:

```swift
let sefr = SEFR()
sefr.fit(examples: X_train, targets: y_train)
let y_pred = sefr.predict(examples: X_test)
```

You can use it for multiclass tasks by using a simple wrapper that performs one-vs-rest:

```swift
let model = SEFRMulticlass()
model.fit(examples: X_train, targets: y_train)
let y_pred = model.predict(examples: X_test)
```

The demo program is a simple Swift script that runs on macOS. It uses the Vision framework to generate "feature prints" for a set of training and test images. Each feature print consists of 2048 numbers. It trains SEFR on these feature prints.

To run the demo program, open a Terminal on macOS and type:

```bash
$ cat SEFR.swift Demo.swift | swift -
```

## Credits

The images used in the dataset were hand-picked from [Google Open Images Dataset V6](https://storage.googleapis.com/openimages/web/index.html). For full credits and license terms, [see here](Image%20Credits.txt).

The source code is [licensed as MIT](LICENSE).

My implementation is based on the following sources:

- [SEFR: A Fast Linear-Time Classifier for Ultra-Low Power Devices](https://arxiv.org/abs/2006.04620)
- [Original Python implementation](https://github.com/sefr-classifier/sefr)
- [Multiclass version in Python](https://github.com/alankrantas/sefr_multiclass_classifier)
