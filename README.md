# Spectrogram Analysiti via self-Attention for Realizing Cross-Model Visual-Audio Generation

Requires TensorFlow>=1.4 and tested with Python 3.5+.

## Preparation of dataset

You should download the Sub-URMP dataset on https://www.cs.rochester.edu/~cxu22/d/vagan/. The sound file should transform to LMS(64x64)

## Train the model using this command:

if you want to trian S2I:

```
python main.py --model S2I
```

Train I2S:

```
python main.py --model I2S
```



## Traing Classifier

The code should pre-trained the classifier first.Our experiment results:

| Pre-trained Classifier | Image | Sound |
| ------ | ------------------ | ----------------------------------- |
| Training | 1.0 | 0.96875 |
| Testing | 0.9375 | 0.90625 |

## Test the model using this command:

If you finish trained your model.You can test your model results by:

S2I

```
python main.py --model S2I
```

I2S

```
python main.py --model I2S
```

## Our results:

| SA-CMGAN | S2I    | I2S    |
| -------- | ------ | ------ |
| Training | 0.9375 | 0.8750 |
| Testing  | 0.8438 | 0.5937 |

