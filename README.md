# Automatic Speech recognition for Speech Assessment of Persian Preschool Children

## Abstract
Preschool evaluation is crucial because it gives teachers and parents influential knowledge about children's growth and development. The COVID-19 pandemic has highlighted the necessity of online assessment for preschool children. One of the areas that should be tested is their ability to speak. Employing an Automatic Speech Recognition(ASR) system is useless since they are pre-trained on voices that are different from children's voices in terms of frequency and amplitude. We constructed an ASR for our cognitive test system to solve this issue using the Wav2Vec 2.0 model with a new pre-training objective called Random Frequency Pitch(RFP). In addition, we used our new dataset to fine-tune our model for Meaningless Words(MW) and Rapid Automatic Naming(RAN) tests. Our new approach reaches a Word Error Rate(WER) of 6.45 on the Persian section of the CommonVoice dataset. Furthermore, our novel methodology produces positive outcomes in zero- and few-shot scenarios.

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

## How to run?
There is a notebook which helps you to run `Wav2Vec 2.0-Base` model. This notebook exists in `Model/Wav2Vec 2.0` directory. Also I suggest you to see the related codes to change wanted parameters.

To run CNN classifier model, you first need to use `Model/VAD` files to create the data needed for the model. Then you can train the model using the following command in `Model/CNN`:
```
python3 main.py
```

## Result

### Fine-tuning Results
| Pre-train Objective | Dataset | WER | Classification Accuracy |
| --- | --- | --- | --- |
| RFP | Common Voice Persian | 6.45 | - |
| Masking | Common Voice Persian | 8.45 | - |
| RFP | LibriSpeech test-clean | 1.35 | - |
| Masking | LibriSpeech test-clean | 1.79 | - |
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

## Datasets
| Name | Address |
| --- | --- |
| Librispeech | [download](https://www.openslr.org/12) |
| CommonVoice Persian | [download](https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-7.0-2021-07-21/cv-corpus-7.0-2021-07-21-fa.tar.gz) |
| Ours | [download](https://drive.google.com/file/d/1clQeyxTurtOu7r39q-CmSmHNapwDDE6u/view?usp=sharing) |

To download `CommonVoice` for other languages, please see <a href="https://github.com/huggingface/datasets/blob/master/datasets/common_voice/common_voice.py">here</a>.


## Citation
```
@article{abaskohi2022automatic,
  title={Automatic Speech recognition for Speech Assessment of Preschool Children},
  author={Abaskohi, Amirhossein and Mortazavi, Fatemeh and Moradi, Hadi},
  journal={arXiv preprint arXiv:2203.12886},
  year={2022}
}
```
