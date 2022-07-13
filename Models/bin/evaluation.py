import argparse

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import soundfile as sf

from layers import Wav2Vec2ForCTC, Wav2Vec2Processor

from datasets import load_metric

from tqdm.auto import tqdm

REQUIRED_SAMPLE_RATE = 16000
AUDIO_MAXLEN, LABEL_MAXLEN = 246000, 256
DO_PADDING = False
USE_CER = False
DATA_DIR = ""

def read_txt_file(f):
	with open(f, "r") as f:
		samples = f.read().split("\n")
		samples = {s.split()[0]: " ".join(s.split()[1:]) for s in samples if len(s.split()) > 2}
	return samples

def read_sound_file(file_path):
	with open(file_path, "rb") as f:
		audio, sample_rate = sf.read(f)
	if sample_rate != REQUIRED_SAMPLE_RATE:
		raise ValueError(
			f"sample rate (={sample_rate}) of your files must be {REQUIRED_SAMPLE_RATE}"
		)
	file_id = os.path.split(file_path)[-1][:-len(".wav")]
	return {file_id: audio}

def tf_forward(speech):
	tf_out = model(speech, training=False)
	return tf.squeeze(tf.argmax(tf_out, axis=-1))

def fetch_sound_text_mapping():
	sound_files = tf.io.gfile.glob(f"{DATA_DIR}/*.wav")
	txt_files = tf.io.gfile.glob(f"{DATA_DIR}/*.txt")

	txt_samples = {}
	for f in txt_files:
		txt_samples.update(read_txt_file(f))

	speech_samples = {}
	for f in sound_files:
		speech_samples.update(read_sound_file(f))

	file_ids = set(speech_samples.keys()) & set(txt_samples.keys())

	print(f"{len(file_ids)} files are found")
	
	samples = [(speech_samples[file_id], txt_samples[file_id]) for file_id in file_ids]
	return samples

def preprocess_text(text):
	label = tokenizer(text)
	label = tf.constant(label, dtype=tf.int32)[None]
	if DO_PADDING:
		label = label[:, :LABEL_MAXLEN]
		padding = tf.zeros((label.shape[0], LABEL_MAXLEN - label.shape[1]), dtype=label.dtype)
		label = tf.concat([label, padding], axis=-1)
	return label

def preprocess_speech(audio):
	audio = tf.constant(audio, dtype=tf.float32)
	audio = processor(audio)[None]
	if DO_PADDING:
		audio = audio[:, :AUDIO_MAXLEN]
		padding = tf.zeros((audio.shape[0], AUDIO_MAXLEN - audio.shape[1]), dtype=audio.dtype)
		audio = tf.concat([audio, padding], axis=-1)
	return audio

def inputs_generator():
	for speech, text in samples:
		yield preprocess_speech(speech), preprocess_text(text)

def infer_librispeech(dataset: tf.data.Dataset, num_batches: int = None):
	predictions, labels = [], []
	for batch in tqdm(dataset, total=num_batches, desc="LibriSpeech Inference ... "):
		speech, label = batch
		tf_out = tf_forward(speech)
		predictions.append(tokenizer.decode(tf_out.numpy().tolist(), group_tokens=True))
		labels.append(tokenizer.decode(label.numpy().squeeze().tolist(), group_tokens=False))
	return predictions, labels

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("model", help="Model dir or model id for HuggingFace🤗")
	parser.add_argument("data", help="Test data path")
	parser.add_argument("audio_max_len", help="Audio max length", default=246000)
	parser.add_argument("lebel_max_len", help="Label max length", default=256)
	parser.add_argument("--padding", help="Turn on output verbosity", action="store_true")
	parser.add_argument("--cer", help="Turn on output verbosity", action="store_true")
	args = parser.parse_args()

	DATA_DIR = args.data
	AUDIO_MAXLEN, LABEL_MAXLEN = args.audio_max_len, args.lebel_max_len
	DO_PADDING = args.padding
	USE_CER = args.cer
	
	model = Wav2Vec2ForCTC.from_pretrained(args.model)

	samples = fetch_sound_text_mapping()

	tokenizer = Wav2Vec2Processor(is_tokenizer=True)
	processor = Wav2Vec2Processor(is_tokenizer=False)

	output_signature = (
		tf.TensorSpec(shape=(None),  dtype=tf.float32),
		tf.TensorSpec(shape=(None), dtype=tf.int32),
	)

	dataset = tf.data.Dataset.from_generator(inputs_generator, output_signature=output_signature)

	predictions, labels = infer_librispeech(dataset, num_batches=2618)

	if USE_CER:
		metric = load_metric("wer")
		metric_name = "WER"
	else:
		metric = load_metric("cer")
		metric_name = "CER"

	print(f"{metric_name}: {metric.compute(references=labels, predictions=predictions)}")