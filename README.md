# Detecting Robot Mistakes in Human-robot Conversations based on Non-verbal Visual Behaviour

This master dissertation investigates the detection of robot mistakes by making use of non-verbal video data. 
It outlines the possibilities and challenges that arise and focusses especially on video data to leverage contextual elements. 
Detecting these mishaps is crucial for enabling smoother conversations and improved robot instructors.

To facilitate this detection, a dataset containing human-robot conversations featuring intentionally designed mistakes in the robot dialogue is employed. 
Natural facial expressions of the participants are used to detect previously made mistakes.
Various machine learning approaches based on an architecture that consists out of a \gls{CNN} followed by an \gls{RNN} are investigated.

While this architecture successfully classifies other datasets with clear emotions, the performance on the main dataset falls short. 
The models consistently overfits on the training dataset without being able to generalize.
Nevertheless, this investigation describes the obstacles and provides guidance on the continuation of working with less expressive and natural data.

To confirm the complexity of the data, a human baseline study is carried out that shows an overall poor accuracy from human annotators and a lack of agreement among them.

# Repository Structure

## LLM:
GPT4-0: test performance with GPT4-0 model on 10 confusion and 10 neutral gifs
performanceLLM: compare flan-t5-large vs gpt-3.5-turbo on Recipe dataset

## Roboflow:
Data
utils.py
5 files for running the different models