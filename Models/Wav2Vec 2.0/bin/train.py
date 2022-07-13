"""
Run this script to launch training

EXAMPLE:
    >>> TPU_NAME=YOUR_TPU BUCKET_NAME=YOUR_BUCKET python3 train.py

"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import asdict, dataclass, field, replace
from typing import List

import tensorflow as tf
import wandb

import numpy as np
from data_utils import LibriSpeechDataLoader, LibriSpeechDataLoaderArgs
from data_utils import CommonVoiceDataLoader, CommonVoiceLoaderArgs
from training_utils import fetch_callbacks, is_gpu_available, is_tpu_available
from layers import CTCLoss, Wav2Vec2Config, Wav2Vec2ForCTC


TPU_NAME = os.getenv("TPU_NAME", "none")
BUCKET_NAME = os.getenv("BUCKET_NAME")
DATA_PATH = os.getenv("DATA_PATH")


@dataclass
class TrainingArgs:
    stage1_lr: float = 1e-3
    stage1_epochs: int = 15

    stage2_lr1: float = 1e-4
    stage2_transition_epochs: int = 10
    stage2_lr2: float = 5e-5
    stage2_epochs: int = 15

    batch_size_per_device: int = 32
    logging_steps: int = 16

    apply_spec_augment: bool = True
    survival_prob: float = 1

    audio_maxlen: int = 246000
    labels_maxlen: int = 256

    seed: int = 42
    from_tfrecords: bool = True

    train_tfrecords: List[str] = field(
        repr=False,
        default_factory=lambda: [
            f"gs://{BUCKET_NAME}/train-clean-100/",
            f"gs://{BUCKET_NAME}/train-clean-360/",
            f"gs://{BUCKET_NAME}/train-other-500/",
        ]
    )

    val_tfrecords: List[str] = field(
        repr=False,
        default_factory=lambda: [
            f"gs://{BUCKET_NAME}/dev-clean/",
        ]
    )

    test_tfrecords: List[str] = field(
        repr=False,
        default_factory=lambda: [
            f"gs://{BUCKET_NAME}/test-clean/",
        ]
    )

    train_dir: str = f"{DATA_PATH}/test-clean/"
    val_dir: str = f"{DATA_PATH}/test-clean/"
    test_dir: str = f"{DATA_PATH}/test-clean/"

    model_id: str = f"gs://{BUCKET_NAME}/tf-wav2vec2-base"
    ckpt_path: str = f"gs://{BUCKET_NAME}/experiment"

    project_name: str = "wav2vec2"

    def __post_init__(self):
        if self.from_tfrecords:
            self.train_dir = self.val_dir = self.test_dir = None

            train_tfrecords = [
                f"{record}*.tfrecord" for record in self.train_tfrecords
            ]
            self.train_tfrecords = tf.io.gfile.glob(train_tfrecords)

            val_tfrecords = [f"{record}*.tfrecord" for record in self.val_tfrecords]
            self.val_tfrecords = tf.io.gfile.glob(val_tfrecords)

            test_tfrecords = [
                f"{record}*.tfrecord" for record in self.test_tfrecords
            ]
            self.test_tfrecords = tf.io.gfile.glob(test_tfrecords)

            assert (
                len(self.train_tfrecords) > 0
                and len(self.val_tfrecords) > 0
                and len(self.test_tfrecords) > 0
            )
        else:
            self.train_tfrecords = self.val_tfrecords = self.test_tfrecords = None


def build_model(args):
    model_config = Wav2Vec2Config(apply_spec_augment=args.apply_spec_augment, survival_prob=args.survival_prob)
    model = Wav2Vec2ForCTC(model_config, input_shape=(1, args.audio_maxlen))
    print(f"loading model from {args.model_id}")
    model.load_weights(f"{args.model_id}/tf_model")
    return model


def train(args):
    if TPU_NAME != "none":
        print("############ INITIATING TPU ############")
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_NAME)
        tf.config.experimental_connect_to_cluster(resolver)
        print("##############################################")

    if is_tpu_available():
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices("TPU"))
        strategy = tf.distribute.TPUStrategy(resolver)
    elif is_gpu_available():
        print("All devices: ", tf.config.list_logical_devices("GPU"))
        strategy = tf.distribute.MirroredStrategy()
    else:
        print("All devices: ", tf.config.list_logical_devices("CPU"))
        raise NotImplementedError

    global_batch_size = strategy.num_replicas_in_sync * args.batch_size_per_device
    print("Training with global batch size of", global_batch_size)
    print(args, end="\n\n")

    print("######### Preparing dataset #########")
    tr_data_args = LibriSpeechDataLoaderArgs(
        data_dir=args.train_dir,
        from_tfrecords=args.from_tfrecords,
        tfrecords=args.train_tfrecords,
        batch_size=global_batch_size,
        audio_maxlen=args.audio_maxlen,
        audio_pad_id=0,
        labels_maxlen=args.labels_maxlen,
        labels_pad_id=0,
    )
    tr_dataset = LibriSpeechDataLoader(tr_data_args)
    tr_dataset = tr_dataset(seed=args.seed, drop_remainder=True)

    val_data_args = replace(
        tr_data_args, data_dir=args.val_dir, tfrecords=args.val_tfrecords
    )
    val_dataset = LibriSpeechDataLoader(val_data_args)
    val_dataset = val_dataset(seed=None, drop_remainder=True)

    test_data_args = replace(
        val_data_args, data_dir=args.test_dir, tfrecords=args.test_tfrecords
    )
    test_dataset = LibriSpeechDataLoader(test_data_args)
    test_dataset = test_dataset(seed=None, drop_remainder=True)

    model_input_shape = (args.batch_size_per_device, args.audio_maxlen)

    with strategy.scope():
        print("######### Preparing model #########")
        model = build_model(args)

        loss = CTCLoss(
            model.config, model_input_shape, division_factor=global_batch_size
        )

        print("######################### STAGE-1 #########################")

        print("######## FREEZING THE BACKBONE (i.e all pretrained weights) ########")

        model.layers[0].trainable = False
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.stage1_lr)
        model.compile(optimizer=optimizer, loss=loss)

        try:
            history = model.fit(
                tr_dataset,
                validation_data=val_dataset,
                epochs=args.stage1_epochs,
                callbacks=fetch_callbacks(args, is_stage2=False),
                verbose="auto",
            )
            print(history.history)
        except KeyboardInterrupt:
            print("Interrupting through KEYBOARD")

        print("###########################################################")

        print("######################### STAGE-2 #########################")

        model.trainable = True
        print("############## FREEZING THE FEATURE_EXTRACTION LAYERS ##############")
        for i in range(len(model.layers[0].layers) - 2):
            model.layers[0].layers[i].trainable = False
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.stage2_lr1)
        model.compile(optimizer=optimizer, loss=loss)

        try:
            history = model.fit(
                tr_dataset,
                validation_data=val_dataset,
                epochs=args.stage2_epochs,
                callbacks=fetch_callbacks(args, is_stage2=True),
                verbose="auto",
            )
            print(history.history)
        except KeyboardInterrupt:
            print("Interrupting through KEYBOARD")

        print("###########################################################")

    print("\n######### Running evaluation #########")
    results = model.evaluate(test_dataset, return_dict=True)
    print(results)


if __name__ == "__main__":
    args = TrainingArgs()
    wandb.init(project=args.project_name, config=asdict(args))
    args.ckpt_path = os.path.join(args.ckpt_path + f"-{wandb.run.id}")

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    train(args)
