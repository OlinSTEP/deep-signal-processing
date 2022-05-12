from typing import Tuple, List, Optional
from argparse import Namespace

import random
from itertools import chain

from .audio_loader import AudioLoader, ThroatMicAudioLoader


class DomainAdaptionAudioLoader(AudioLoader):
    """AudioLoader that produces splits for finetuning and domain adaption"""

    def __init__(self, config: Namespace) -> None:
        self.da_sessions: int = config.da_sessions

        if config.split_subjects != 1 or config.split_sessions == 1:
            raise ValueError(
                "Domain adaption only supports subject splitting"
            )

        super().__init__(config)

    def build_splits(self) -> Tuple[
        List[int], List[int], List[int], List[int], List[int]
    ]:
        """
        Builds train / dev / test split indexs

        :rtype Tuple[
            List[int], List[int], List[int], List[int], List[int]
        ]: Tuple of train / dev train / dev / test train / test split idxs,
        where each list containst the indicies for items in the repspective
        split. dev train and test train splits are for finetuning
        """
        num_subjects = len(self.subjects)
        train_subjects = int(num_subjects * self.train_split)
        test_subjects = int(num_subjects * self.test_split)
        dev_subjects = num_subjects - train_subjects -  test_subjects

        random.seed(self.seed)
        subject_idxs = list(range(num_subjects))
        random.shuffle(subject_idxs)

        def _load_subject_idxs(
            dest: List[int], n: int, dest_train: Optional[List[int]] = None
        ) -> None:
            """Adds file idxs to dest. Optionally adds self.da_sessions sessions
            worth of files to dest_train"""
            if n == 0:
                return

            # Get subject idxs to use for this split
            _subject_idxs = [subject_idxs.pop() for _ in range(n)]
            # Get session idxs to use for this split (not flat)
            session_idxs = [self.subjects[i] for i in _subject_idxs]
            # Flatten sesison idxs
            session_idxs = list(chain(*session_idxs))
            # Get file idxs to use for this split (not flat)
            files = [self.sessions[i] for i in session_idxs]

            if dest_train is not None:
                for _ in range(self.da_sessions):
                    dest_train.extend(files.pop())
            for _ in range(len(files)):
                dest.extend(files.pop())

        train_idxs = []
        dev_train_idxs, dev_test_idxs = [], []
        test_train_idxs, test_test_idxs = [], []

        _load_subject_idxs(train_idxs, train_subjects)
        _load_subject_idxs(dev_test_idxs, dev_subjects, dev_train_idxs)
        _load_subject_idxs(test_test_idxs, test_subjects, test_train_idxs)

        self.train_idxs = set(train_idxs + dev_train_idxs + test_train_idxs)
        return (
            train_idxs,
            dev_train_idxs, dev_test_idxs,
            test_train_idxs, test_test_idxs
        )


class ThroatMicDomainAdaptionAudioLoader(
    DomainAdaptionAudioLoader, ThroatMicAudioLoader
):
    pass
