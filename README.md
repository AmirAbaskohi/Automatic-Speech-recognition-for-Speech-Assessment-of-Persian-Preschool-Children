# Automatic Speech recognition for Speech Assessment of Persian Preschool Children

## Abstract
Preschool evaluation is crucial because it gives teachers, parents, and families crucial knowledge about a child's growth and development. The pandemic has highlighted the necessity for preschool children to be assessed online. This online testing requires a variety of technologies, from web application development to various artificial intelligence models in diverse criteria such as speech recognition. Because of the acoustic fluctuations and differences in voice frequencies between children and adults, employing Automatic Speech Recognition(ASR) systems is difficult because they are pre-trained on adults' voices. In addition, training a new model requires a large amount of data. To solve this issue, we constructed an ASR for our cognitive test system using the Wav2Vec 2.0 model with a new pre-training objective, called Random Frequency Pitch(RFP), and our new dataset, which was tested on Meaningless Words(MW) and Rapid Automatic Naming(RAN) tests. Due to the peculiarities of these two tests, we explored numerous models, including Convolutional Neural Network (CNN) and Wav2Vec 2.0 models. Our new approach, reaches Word Error Rate(WER) of 6.45 on the Persian section of CommonVoice dataset. Furthermore our novel methodology produces positive outcomes in zero-shot and few-shot scenarios.

## Assessments
In our system, ASR is required for two tests. The first test is called Rapid Automatic Naming(RAN). The RAN task is
a behavioral test that evaluates how fast and accurately people name groups of visual stimuli. For fluent naming of
a sequence of visual stimuli, RAN depends on the coordination of various processes into a synchronized access and
retrieval mechanism. The difficulty in this activity includes a word sequence, as well as the necessity of
correctness because the findings will be used to assess children’s ability to talk. In addition, due to the importance of
speed in this activity, the model should provide accurate high-speed utterances.

![image](https://user-images.githubusercontent.com/50926437/156012200-15510ead-f03c-4344-bb6f-10170bc12582.png)

The second test is a phonological memory test, in which youngsters are asked to listen to a meaningless word and
then repeat it. This exam is significant complex because it demonstrates strong developmental connections between
test results and vocabulary, reading, and overall abilities in young children. The task’s primary difficulty is the
great degree of similarity between these words and the actual words. For example "mashogh" is like "ghashogh"(spoon)
except the first letter and the first phoneme is different. These similarities make classification hard. It is crucial to note that accuracy is essential in this test since it focuses on the children’s listening skills, and they must repeat what they hear in the test rather than the term that exists in reality.

## Dataset

We collected our own dataset from children which details are described in the paper. You can find our dataset <a href="https://drive.google.com/file/d/1clQeyxTurtOu7r39q-CmSmHNapwDDE6u/view?usp=sharing">here</a>.

Dataset Details:

| Color      | Number of Data Gathered |
| ----------- | ----------- |
| Blue      | 483       |
| Red   | 482        |
| Black   | 873        |
| Green   | 472        |
| Yellow   | 488        |

| Environment      | Number of Data Gathered |
| ----------- | ----------- |
| Clean      | 90       |
| Noisy   | 24        |

## Models
Three models are available in this repository:
* VAD + Classifier

![image](https://user-images.githubusercontent.com/50926437/156013625-467b48b2-0f9d-4d55-aeb4-ae16e6a141b5.png)

![image](https://user-images.githubusercontent.com/50926437/156013632-2e3e97ae-b867-4f5e-bf09-a1fce7bd821e.png)

* Wav2Vec 2.0

![image](https://user-images.githubusercontent.com/50926437/156013711-dfafdd0d-7670-45a5-bbf2-127fb416f94b.png)

## Result

### Fine-tuning Results
| Pre-train Objective | Dataset | WER | Classification Accuracy |
| --- | --- | --- | --- |
| RFP | Common Voice | 6.45 | --- |
| Masking | Common Voice | 8.45 | --- |
| --- | --- | --- | --- |
| RFP | Common Voice + RAN's Samples | 4.56 | 0.98 |
| Masking | Common Voice + RAN's Samples | 5.15 | 0.87 |
| RFP | Common Voice + Meaningless Words's Samples | 4.12 | 0.99 |
| Masking | Common Voice + Meaningless Words's Samples | 7.15 | 0.84 |

### Zero and Few-shot Results
| Fine-tuning Steps | RFP | Masking |
| --- | --- | --- |
| 0 | 37.86 | 42.29 |
| 10k | 30.48 | 33.75 |
| 20k | 26.65 | 31.18 |
| 30k | 13.12 | 15.12 |
| 40k | 11.29 | 14.68 |
| 50k | 10.41 | 12.49 |
| zero-shot | 37.86 | 42.29 |

## Checkpoints
| Model | Address |
| --- | --- |
| Fine-tuned | [download](https://drive.google.com/drive/folders/1-U9-ClJQv0pQuiAfxQp38U4GYs_vhL1R?usp=sharing) |
| Mask Pre-trained | [download](https://drive.google.com/drive/folders/1-3KmmvLi3HtTsZLd5dVA9ZhNhd5PL5T5?usp=sharing) |
| RFP Pre-trained | [download](https://drive.google.com/drive/folders/1-YiTt5KHcGsxircAMm2qcW9T3r2QhmD7?usp=sharing) |


## Citation
```
@article{abaskohi2022automatic,
  title={Automatic Speech recognition for Speech Assessment of Preschool Children},
  author={Abaskohi, Amirhossein and Mortazavi, Fatemeh and Moradi, Hadi},
  journal={arXiv preprint arXiv:2203.12886},
  year={2022}
}
```
