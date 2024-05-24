# Detecting Robot Mistakes in Human-robot Conversations based on Non-verbal Visual Behaviour

This master dissertation investigates the detection of robot mistakes by making use of non-verbal video data on human-robot conversations. 
Detecting these mishaps is crucial for enabling smoother conversations and improved robot instructors.
To facilitate this detection, a dataset containing human-robot conversations featuring intentionally designed mistakes in the robot dialogue is employed. 
Natural facial expressions of the participants are used to detect previously made mistakes, with the focus on video data to leverage contextual elements.
Various Machine Learning (ML) approaches based on an architecture that consists out of a Convolution Neural Network (CNN) followed by a Recurrent Neural Network (RNN) are investigated.
While this architecture successfully classifies other datasets with clear and unambiguous emotional expressions, it is not capable of capturing information from the natural expressions as a response to robot mistakes.
To confirm the complexity of the data, a human baseline study is carried out that shows an overall poor accuracy from human annotators and a lack of agreement among them.
This dissertation describes the obstacles and provides guidance on the continuation of working with less expressive and natural data.

**Keywords**: Robot Mistakes, Non-verbal video data, Human-Robot Conversation

# Repository Structure

To install the corresponding libraries and depencencies use

`pip install -r requirements.txt`

or with conda:

```
conda env create -f environment.yml
conda activate ThesisRobot
```

## LLM:
Do not rerun these files as they are based on unavailable data.

GPT4-0: test performance with GPT4-0 model on 10 confusion and 10 neutral gifs

performanceLLM: compare Flan-t5-large vs gpt-3.5-turbo on Recipe dataset

## Roboflow:
The `data` folder contains the necessary data to run various models, and `utils.py` contains function that are used by other files.

There are 5 interactive notebooks for running the 5 different models.

## BlendshapeAnalysis:
Do not rerun these files as they are based on unavailable data.

See Section 5.5 Statistical Tests

## GIF:
The files on this folders can by used directly as a demo.
The data is saved in `data`, `blendshape`, and `keypoints` and the output of the training procedures will be in `pickled`, `models`, and `images`.
The other 3 directories each consist out of 4 files:

- testXGIF.ipynb: For testing purposes
- createXDataset.py: For creating a dataset based on the amount of skipframes
- runXGIF.py: For training, plotting and saving models.
- utils.py: For functions and classes used by other files.

To reproduce the results from the Master Thesis, the following commands can be executed.
For example for the Image based model:

```
python createImageDataset.py -s <skipframes>
python runImageGIFModel.py -b <batch_size> -s <skipframes> -e <epoch> -lr <learning_rate> -pos <positive_weight> -w <workers> -is <input_size> -hs <hidden_size> -nl <num_layers> -name <name> -bb <backbone> -gpu <gpus>
```

Afterwards, the dataset will be saved in `pickled`, the model in `models` and the train and validation curves, the ROC-curves and the validation matrix will be saved in `images`.


## Recipe:

These files have similar functionality to the GIF dataset, but due to privacy reason they cannot be used and have not been altered to work out of the box.

## ReleventMomentExtraction

This investigates 20 arbitrary chosen videos with annotated relevant moments. 
It predicts relevant moments based on blendshape and keypoint peak detection algorithms.

