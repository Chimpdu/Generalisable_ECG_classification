import ast
import random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import wfdb
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

def safe_literal_eval(x):
    # Use: convert a string like "{'NORM': 100, 'MI': 50}" into a Python dict. The original data is stored in a string format.
    # Input: x: string like "{'NORM': 100, 'MI': 50}"
    # Output: dict: {'NORM': 100, 'MI': 50}
   
    if pd.isna(x):
        return np.nan
    return ast.literal_eval(x)

def load_ptb_metadata(ptb_root):
    # Use: load ptb-xl metadata from ptbxl_database.csv file, the csv file is provided in the ptb-xl dataset.
    # Input: ptb_root: Path to the ptb-xl dataset root directory.
    # Output: ptb_df: DataFrame containing the ptb-xl metadata.
    ptb_df = pd.read_csv(ptb_root / "ptbxl_database.csv")

    # Parse SCP code dictionary, this is the label of ptb dataset.
    ptb_df["scp_codes_dict"] = ptb_df["scp_codes"].apply(safe_literal_eval)

    # Count how many SCP codes each ECG has
    ptb_df["n_scp_codes"] = ptb_df["scp_codes_dict"].apply(
        lambda d: len(d) if isinstance(d, dict) else np.nan
    )

    # Create paths for 100 Hz and 500 Hz records, help future retrieval
    ptb_df["record_base_100"] = ptb_df["filename_lr"].apply(lambda x: str(ptb_root / x))
    ptb_df["record_base_500"] = ptb_df["filename_hr"].apply(lambda x: str(ptb_root / x))

    return ptb_df

def print_basic_df_info(df, name, n_rows = 5):
    # Use: print the basic information of a dataframe.
    # Input: df: DataFrame to print, name: name of the dataframe, n_rows: number of rows to print.
    # Output: pure side effect.

    print("_" * 70)
    print(name)
    print("_" * 70)
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nDtypes:")
    print(df.dtypes)
    print(f"\nFirst {n_rows} rows:")
    print(df.head(n_rows))
    print()

def missingness_report(df, record_base_col, dataset_name = "dataset"):
    # Use: compute and plot waveform-level missingness for each record.
    # Input: df: DataFrame containing the records, record_base_col: column containing the record base paths, dataset_name: name of the dataset.
    # Output: record_report: DataFrame containing the waveform-level missingness for each record, channel_stats: DataFrame containing the channel-level stats, channel_level_df: DataFrame containing the record-channel pair stats.

    record_rows = []
    channel_rows = []

    # compute waveform-level missingness for each record
    for record_base in df[record_base_col]:
        record = wfdb.rdrecord(record_base)
        signal = record.p_signal                     # shape: [n_samples, n_channels]
        sig_names = record.sig_name

        n_samples, n_channels = signal.shape
        total_values = signal.size
        n_missing = int(np.isnan(signal).sum())
        pct_missing = 100.0 * n_missing / total_values

        record_rows.append({
            "dataset": dataset_name,
            "record_base": record_base,
            "n_samples": n_samples,
            "n_channels": n_channels,
            "total_values": total_values,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "channel_names": sig_names,
        })

        # Record-channel level stats
        for ch_idx, ch_name in enumerate(sig_names):
            x = signal[:, ch_idx]
            ch_total = len(x)
            ch_missing = int(np.isnan(x).sum())
            ch_pct_missing = 100.0 * ch_missing / ch_total

            channel_rows.append({
                "dataset": dataset_name,
                "record_base": record_base,
                "channel_name": ch_name,
                "n_samples": ch_total,
                "n_missing": ch_missing,
                "pct_missing": ch_pct_missing,
                "has_missing": int(ch_missing > 0),
            })

    record_report = pd.DataFrame(record_rows).sort_values(
        ["pct_missing", "record_base"], ascending=[False, True]
    ).reset_index(drop=True)

    channel_level_df = pd.DataFrame(channel_rows)

    channel_stats = (
        channel_level_df.groupby("channel_name", as_index=False)
        .agg(
            channel_count=("channel_name", "size"),
            total_samples=("n_samples", "sum"),
            total_missing=("n_missing", "sum"),
            n_record_channels_with_missing=("has_missing", "sum"),
        )
    )

    channel_stats["pct_missing"] = (
        100.0 * channel_stats["total_missing"] / channel_stats["total_samples"]
    )

    channel_stats = channel_stats.sort_values(
        ["channel_count", "channel_name"], ascending=[False, True]
    ).reset_index(drop=True)

  
    print("_" * 70)
    print(f"{dataset_name} waveform missingness report")
    print("_" * 70)
    print(record_report.head(10))
    print()

    print("_" * 70)
    print(f"{dataset_name} channel stats")
    print("_" * 70)
    print(channel_stats)
    print()

    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=channel_stats, x="channel_name", y="channel_count")
    plt.title(f"{dataset_name} channel counts")
    plt.xlabel("Channel name")
    plt.ylabel("Count across records")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    heatmap_df = channel_level_df.pivot(
        index="record_base",
        columns="channel_name",
        values="pct_missing"
    ).fillna(0.0)

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_df,
        yticklabels=False,
        cmap="viridis",
        cbar=False
    )
    plt.title(f"{dataset_name} waveform missingness heatmap")
    plt.xlabel("Channels")
    plt.ylabel("Records")
    plt.tight_layout()
    plt.show()

    return record_report, channel_stats, channel_level_df




def load_ptb_scp_mapping(ptb_root):
    # Use: load ptb-xl scp statement mapping table from scp_statements.csv file, the csv file is provided in the ptb-xl dataset.
    # Input: ptb_root: Path to the ptb-xl dataset root directory.
    # Output: scp_df: DataFrame containing the ptb-xl scp statement mapping table.
    scp_df = pd.read_csv(ptb_root / "scp_statements.csv", index_col=0)
    return scp_df


def add_ptb_superclass_targets(ptb_df, scp_df):
    # Use: add ptb-xl 5 diagnostic class targets: NORM, MI, STTC, CD, HYP to the ptb-xl metadata dataframe.
    # Input: ptb_df: DataFrame containing the ptb-xl metadata, scp_df: DataFrame containing the ptb-xl scp statement mapping table.
    # Output: ptb_df: DataFrame containing the ptb-xl metadata with the 5 diagnostic superclass targets.
    # We predict the 5 diagnostic superclass.
    ptb_df = ptb_df.copy()

    diagnostic_codes = scp_df[scp_df["diagnostic"] == 1].copy()
    valid_superclasses = ["NORM", "MI", "STTC", "CD", "HYP"]

    for cls in valid_superclasses:
        ptb_df[cls] = 0

    def get_superclasses_from_scp_dict(scp_dict):
        if not isinstance(scp_dict, dict):
            return []

        found = set()
        for code in scp_dict.keys():
            if code in diagnostic_codes.index:
                diag_class = diagnostic_codes.loc[code, "diagnostic_class"]
                if pd.notna(diag_class) and diag_class in valid_superclasses:
                    found.add(diag_class)

        return list(found)

    ptb_df["superclasses"] = ptb_df["scp_codes_dict"].apply(get_superclasses_from_scp_dict)

    for cls in valid_superclasses:
        ptb_df[cls] = ptb_df["superclasses"].apply(lambda xs: int(cls in xs))

    ptb_df["n_superclasses"] = ptb_df[valid_superclasses].sum(axis=1)

    return ptb_df



def get_ltdb_record_ids(ltdb_root):
    # Use: get the record ids of the LTDB dataset.
    # Input: ltdb_root: Path to the LTDB dataset root directory.
    # Output: record_ids: list of record ids.
    with open(ltdb_root / "RECORDS", "r") as f:
        record_ids = [line.strip() for line in f if line.strip()]
    return record_ids


def summarize_ltdb_record(ltdb_root, record_id):
    # Use: summarize one LTDB record.
    # Input: ltdb_root: Path to the LTDB dataset root directory, record_id: id of the record.
    # Output: summary: dictionary containing the summary of the record.
    record_base = str(ltdb_root / record_id)

    record = wfdb.rdrecord(record_base)
    ann = wfdb.rdann(record_base, "atr")

    ann_symbols = ann.symbol
    ann_counts = Counter(ann_symbols)

    summary = {
        "record_id": record_id,
        "record_base": record_base,
        "fs": record.fs,
        "sig_len": record.sig_len,
        "duration_sec": record.sig_len / record.fs,
        "duration_min": (record.sig_len / record.fs) / 60.0,
        "n_channels": record.n_sig,
        "sig_names": record.sig_name,
        "units": record.units,
        "comments": record.comments,
        "n_annotations": len(ann_symbols),
        "unique_annotation_symbols": len(set(ann_symbols)),
        "top_annotation_symbols": dict(ann_counts.most_common(10)),
    }

    return summary


def build_ltdb_summary_df(ltdb_root):
    # Use: build a summary dataframe for the LTDB dataset.
    # Input: ltdb_root: Path to the LTDB dataset root directory.
    # Output: ltdb_df: DataFrame containing the summary of the LTDB dataset.
    record_ids = get_ltdb_record_ids(ltdb_root)
    rows = [summarize_ltdb_record(ltdb_root, rid) for rid in record_ids]
    ltdb_df = pd.DataFrame(rows)
    return ltdb_df



def load_ecg_window(record_base, start_sec = 0.0, seconds = 10.0):
    # Use: load a short ECG segment from a WFDB record.
    # Input: record_base: Path to the WFDB record, start_sec: start time in seconds, seconds: duration in seconds.
    # Output: signal: [n_samples, n_channels], fs: sampling frequency, sig_names: names of the channels.
    header = wfdb.rdheader(record_base)
    fs = header.fs

    sampfrom = int(start_sec * fs)
    sampto = int((start_sec + seconds) * fs)

    record = wfdb.rdrecord(record_base, sampfrom=sampfrom, sampto=sampto)
    signal = record.p_signal
    sig_names = record.sig_name

    return signal, fs, sig_names



def plot_ecg_record(record_base, title = "", start_sec = 0.0, seconds = 10.0):
    # Use: plot all channels of a short ECG segment.
    # Input: record_base: Path to the WFDB record, title: title of the plot, start_sec: start time in seconds, seconds: duration in seconds.
    # Output: pure side effect.
    signal, fs, sig_names = load_ecg_window(record_base, start_sec=start_sec, seconds=seconds)

    n_samples, n_channels = signal.shape
    time = np.arange(n_samples) / fs + start_sec

    plt.figure(figsize=(14, 6))
    for ch in range(n_channels):
        plt.plot(time, signal[:, ch], label=sig_names[ch])

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title if title else f"ECG plot: {Path(record_base).name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_ecg_with_annotations(record_base, ann_ext = "atr", start_sec = 0.0, seconds = 10.0):
    # Use: plot first channel and overlay annotation markers. Very useful for LTDB exploration.
    # Input: record_base: Path to the WFDB record, ann_ext: extension of the annotation file, start_sec: start time in seconds, seconds: duration in seconds.
    # Output: pure side effect.
    signal, fs, sig_names = load_ecg_window(record_base, start_sec=start_sec, seconds=seconds)
    ann = wfdb.rdann(record_base, ann_ext)

    n_samples = signal.shape[0]
    sampfrom = int(start_sec * fs)
    sampto = int((start_sec + seconds) * fs)
    time = np.arange(n_samples) / fs + start_sec

    plt.figure(figsize=(14, 6))
    plt.plot(time, signal[:, 0], label=sig_names[0])

    for s, sym in zip(ann.sample, ann.symbol):
        if sampfrom <= s < sampto:
            local_idx = s - sampfrom
            x = s / fs
            y = signal[local_idx, 0]
            plt.axvline(x=x, linestyle="--", alpha=0.4)
            plt.text(x, y, sym, fontsize=8)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(f"{Path(record_base).name} with annotations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_ptb_metadata_overview(ptb_df):
    # Use: plot simple PTB-XL metadata distributions.
    # Input: ptb_df: DataFrame containing the PTB-XL metadata.
    # Output: pure side effect.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(ptb_df["age"].dropna(), bins=30)
    axes[0, 0].set_title("PTB-XL Age Distribution")
    axes[0, 0].set_xlabel("Age")
    axes[0, 0].set_ylabel("Count")

    ptb_df["sex"].value_counts(dropna=False).sort_index().plot(kind="bar", ax=axes[0, 1])
    axes[0, 1].set_title("PTB-XL Sex Distribution")
    axes[0, 1].set_xlabel("Sex")
    axes[0, 1].set_ylabel("Count")

    ptb_df["strat_fold"].value_counts().sort_index().plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title("PTB-XL Stratified Fold Counts")
    axes[1, 0].set_xlabel("strat_fold")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(ptb_df["n_scp_codes"].dropna(), bins=20)
    axes[1, 1].set_title("PTB-XL Number of SCP Codes per Record")
    axes[1, 1].set_xlabel("n_scp_codes")
    axes[1, 1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()


def plot_ptb_superclass_counts(ptb_df):
    # Use: plot counts for PTB-XL 5 superclass labels.
    # Input: ptb_df: DataFrame containing the PTB-XL metadata.
    # Output: pure side effect.
    superclass_cols = ["NORM", "MI", "STTC", "CD", "HYP"]
    counts = ptb_df[superclass_cols].sum().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("PTB-XL Diagnostic Superclass Counts")
    plt.xlabel("Superclass")
    plt.ylabel("Number of records with label")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def plot_ltdb_record_overview(ltdb_df):
    # Use: plot LTDB duration and annotation count per record.
    # Input: ltdb_df: DataFrame containing the LTDB metadata.
    # Output: pure side effect.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(ltdb_df["record_id"], ltdb_df["duration_min"])
    axes[0].set_title("LTDB Record Duration (minutes)")
    axes[0].set_xlabel("Record ID")
    axes[0].set_ylabel("Duration (minutes)")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(ltdb_df["record_id"], ltdb_df["n_annotations"])
    axes[1].set_title("LTDB Number of Annotations per Record")
    axes[1].set_xlabel("Record ID")
    axes[1].set_ylabel("Annotation count")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


    #################### unchecked
def zscore_1d(x, eps=1e-6):
    # Use: z-score one channel. Normalize the channel to have mean 0 and standard deviation 1.
    # Input: x: [T]
    # Output: [T]
    x = x.astype(np.float32)
    return ((x - np.mean(x)) / (np.std(x) + eps)).astype(np.float32)





def crop_or_pad_1d(x, target_len, random_crop=True):
    # Use: make the length of the input signal equal to the target length.
    # Input: x: [T], target_len: target length.
    # Output: [target_len]
    T = len(x)

    if T == target_len:
        return x.astype(np.float32)

    if T > target_len:
        if random_crop:
            start = np.random.randint(0, T - target_len + 1)
        else:
            start = 0
        return x[start:start + target_len].astype(np.float32)

    out = np.zeros((target_len,), dtype=np.float32)
    out[:T] = x
    return out





def resample_1d(x, orig_fs, target_fs=100):
    # Resample a channel so that all channels have the same sampling frequency.
    # Input: x: [T], orig_fs: original sampling frequency, target_fs: target sampling frequency.
    # Output: [T]
    if int(orig_fs) == int(target_fs):
        return x.astype(np.float32)
    return resample_poly(x, up=target_fs, down=int(orig_fs)).astype(np.float32)





def bandpass_filter_1d(x, fs, low=0.5, high=40.0, order=4):
    # Use: this function is optional but it allows user to apply a bandpass filter to the signal. Turned out to be usefull for the encoder training.
    # Input: x: [T], fs: sampling frequency, low: low cutoff frequency, high: high cutoff frequency, order: order of the filter.
    # Output: [T]
 
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    sos = butter(order, [low_n, high_n], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)



def make_ptb_splits(ptb_df):
    # Use: split the PTB-XL dataset into train, validation and test sets according to the official split provided by the PTB-XL dataset.
    # Input: ptb_df: DataFrame containing the PTB-XL metadata.
    # Output: train_df: DataFrame containing the train set, val_df: DataFrame containing the validation set, test_df: DataFrame containing the test set.
    train_df = ptb_df[ptb_df["strat_fold"].between(1, 8)].reset_index(drop=True)
    val_df   = ptb_df[ptb_df["strat_fold"] == 9].reset_index(drop=True)
    test_df  = ptb_df[ptb_df["strat_fold"] == 10].reset_index(drop=True)
    return train_df, val_df, test_df






def make_ltdb_split(ltdb_record_ids, seed=42):
    # Use: split the LTDB dataset into train, validation and test sets according to a fixed split. 5 train, 1 validation, 1 test.
    # Input: ltdb_record_ids: list of record ids, seed: random seed.
    # Output: train_ids: list of train record ids, val_ids: list of validation record ids, test_ids: list of test record ids.

    ids = list(ltdb_record_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    train_ids = ids[:5]
    val_ids   = [ids[5]]
    test_ids  = [ids[6]]

    return train_ids, val_ids, test_ids



def load_full_record(record_base):
    # Use:load a entire ECG record.
    # Input: record_base: Path to the WFDB record.
    # Output: signal: [T, C], fs: sampling frequency, sig_names: names of the channels.
    rec = wfdb.rdrecord(record_base)
    return rec.p_signal.astype(np.float32), rec.fs, rec.sig_name


def load_record_segment(record_base, start_sample=None, end_sample=None):
    # Use: load a segment from an ECG record.
    # Input: record_base: Path to the WFDB record, start_sample: start sample, end_sample: end sample.
    # Output: signal: [T, C], fs: sampling frequency, sig_names: names of the channels.
    if start_sample is None and end_sample is None:
        rec = wfdb.rdrecord(record_base)
    else:
        rec = wfdb.rdrecord(record_base, sampfrom=start_sample, sampto=end_sample)

    return rec.p_signal.astype(np.float32), rec.fs, rec.sig_name



def preprocess_single_channel(
    x_t,
    orig_fs,
    target_fs=100,
    target_len=500,
    use_filter=False,
    random_crop=True,
):
    # Use: preprocess one channel for encoder pretraining. It does filtering (optional), resampling, z-score normalization, cropping/padding, and adding channel dimension.
    # Input: x_t: [T], orig_fs: original sampling frequency, target_fs: target sampling frequency, target_len: target length, use_filter: whether to use filtering, random_crop: whether to randomly crop the signal.
    # Output: x: [1, target_len]
    x = x_t.astype(np.float32)

    # filter (off by default)
    if use_filter:
        x = bandpass_filter_1d(x, fs=orig_fs, low=0.5, high=40.0, order=4)

    # resample
    x = resample_1d(x, orig_fs=orig_fs, target_fs=target_fs)

    # normalize
    x = zscore_1d(x)

    # crop/pad
    x = crop_or_pad_1d(x, target_len=target_len, random_crop=random_crop)

    # [1, T]
    return x[None, :].astype(np.float32)



def make_time_patch_mask(batch_x, patch_size=25, mask_ratio=0.5):
    # Use: create masks for training the encoder.
    # Input: batch_x: [B, 1, T], patch_size: size of the patch, mask_ratio: ratio of the patches to be masked.
    # Output: patch_mask: [B, N_patches], True means masked.
    B, C, T = batch_x.shape
    assert C == 1, "Expected single-channel input for pretraining"
    assert T % patch_size == 0, "T must be divisible by patch_size"

    n_patches = T // patch_size
    n_mask = int(round(mask_ratio * n_patches))

    patch_mask = torch.zeros((B, n_patches), dtype=torch.bool)

    for b in range(B):
        idx = torch.randperm(n_patches)[:n_mask]
        patch_mask[b, idx] = True

    return patch_mask


def mae_collate_fn(batch, patch_size=25, mask_ratio=0.5):
    # Use: collate function (for DataLoader) for training the encoder.
    # Input: batch: list of dictionaries containing the data, patch_size: size of the patch, mask_ratio: ratio of the patches to be masked.
    # Output: dictionary containing the data, patch_mask: [B, N_patches], True means masked.
    x = torch.stack([item["x"] for item in batch], dim=0)
    patch_mask = make_time_patch_mask(x, patch_size=patch_size, mask_ratio=mask_ratio)

    return {
        "x": x,
        "patch_mask": patch_mask,
        "source": [item["source"] for item in batch],
        "channel_name": [item["channel_name"] for item in batch],
        "record_base": [item["record_base"] for item in batch],
    }



class PTBPretrainSingleChannelDataset(Dataset):
    # Use: dataset for training the encoder on PTB-XL. Each sample is one (record, channel) pair.
    # Input: ptb_train_df: DataFrame containing the PTB-XL train set, target_fs: target sampling frequency, seconds: duration of the signal, use_filter: whether to use filtering.
    # Output: pure side effect.
    def __init__(self, ptb_train_df, target_fs=100, seconds=5.0, use_filter=False):
        self.df = ptb_train_df.reset_index(drop=True)
        self.target_fs = target_fs
        self.target_len = int(target_fs * seconds)
        self.use_filter = use_filter

        # Build an index of (row_idx, channel_idx)
        self.samples = []
        for row_idx in range(len(self.df)):
            # PTB records100 always have 12 channels, but we read the header to be safe
            record_base = self.df.iloc[row_idx]["record_base_100"]
            header = wfdb.rdheader(record_base)
            n_channels = header.n_sig

            for ch_idx in range(n_channels):
                self.samples.append((row_idx, ch_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row_idx, ch_idx = self.samples[idx]

        row = self.df.iloc[row_idx]
        record_base = row["record_base_100"]

        signal_tc, fs, sig_names = load_full_record(record_base)   # [T, C]
        x_t = signal_tc[:, ch_idx]                                 # [T]

        x = preprocess_single_channel(
            x_t,
            orig_fs=fs,
            target_fs=self.target_fs,
            target_len=self.target_len,
            use_filter=self.use_filter,
            random_crop=True,   # random 5-second crop
        )

        return {
            "x": torch.from_numpy(x),      # [1, T]
            "source": "PTB",
            "channel_name": sig_names[ch_idx],
            "record_base": record_base,
            "row_idx": row_idx,
            "channel_idx": ch_idx,
        }

class LTDBPretrainSingleChannelDataset(Dataset):
    # Use: dataset for training the encoder on LTDB. Each sample is one (record, channel) pair.
    # Input: ltdb_root: Path to the LTDB dataset root directory, train_record_ids: list of record ids, target_fs: target sampling frequency, seconds: duration of the signal, samples_per_epoch: number of samples per epoch, use_filter: whether to use filtering.
    # Output: pure side effect.
    def __init__(self, ltdb_root, train_record_ids, target_fs=100, seconds=5.0, samples_per_epoch=None, use_filter=False):
        self.root = ltdb_root
        self.record_ids = list(train_record_ids)
        self.target_fs = target_fs
        self.target_len = int(target_fs * seconds)
        self.seconds = seconds
        self.use_filter = use_filter

        # Build an index of all real LTDB channels
        self.channel_sources = []
        for rid in self.record_ids:
            record_base = str(self.root / rid)
            header = wfdb.rdheader(record_base)
            n_channels = header.n_sig
            sig_names = header.sig_name
            fs = header.fs
            sig_len = header.sig_len

            for ch_idx in range(n_channels):
                self.channel_sources.append({
                    "record_id": rid,
                    "record_base": record_base,
                    "channel_idx": ch_idx,
                    "channel_name": sig_names[ch_idx],
                    "fs": fs,
                    "sig_len": sig_len,
                })

        # If not specified, one epoch = one pass over all record-channel sources
        self.samples_per_epoch = samples_per_epoch or len(self.channel_sources)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Cycle through all channel sources
        source = self.channel_sources[idx % len(self.channel_sources)]

        record_base = source["record_base"]
        ch_idx = source["channel_idx"]
        fs = source["fs"]
        sig_len = source["sig_len"]

        raw_len = int(self.seconds * fs)
        max_start = max(0, sig_len - raw_len)
        start = random.randint(0, max_start)

        signal_tc, fs, sig_names = load_record_segment(
            record_base,
            start_sample=start,
            end_sample=start + raw_len
        )

        x_t = signal_tc[:, ch_idx]

        x = preprocess_single_channel(
            x_t,
            orig_fs=fs,
            target_fs=self.target_fs,
            target_len=self.target_len,
            use_filter=self.use_filter,
            random_crop=False,   # window already selected
        )

        return {
            "x": torch.from_numpy(x),      # [1, T]
            "source": "LTDB",
            "channel_name": source["channel_name"],
            "record_base": record_base,
            "record_id": source["record_id"],
            "channel_idx": ch_idx,
        }


class BalancedPretrainDataset(Dataset):
    # Use: dataset for training the encoder on PTB-XL and LTDB. It mixes the samples from PTB-XL and LTDB with a fixed probability.
    # Input: ptb_ds: dataset for training the encoder on PTB-XL, ltdb_ds: dataset for training the encoder on LTDB, p_ptb: probability of selecting a PTB-XL sample, samples_per_epoch: number of samples per epoch.
    # Output: pure side effect.
    def __init__(self, ptb_ds, ltdb_ds, p_ptb=0.5, samples_per_epoch=8000):
        self.ptb_ds = ptb_ds
        self.ltdb_ds = ltdb_ds
        self.p_ptb = p_ptb
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if random.random() < self.p_ptb:
            return self.ptb_ds[random.randrange(len(self.ptb_ds))]
        else:
            return self.ltdb_ds[random.randrange(len(self.ltdb_ds))]




def patchify_1d(x, patch_size=25):
    # Use: split the signal into patches. Same patch size as used in the mask creation.
    # Input: x: [B, 1, T], patch_size: size of the patch.
    # Output: patches: [B, N_patches, patch_size].
    B, C, T = x.shape
    assert C == 1, "Expected single-channel input"
    assert T % patch_size == 0, "T must be divisible by patch_size"

    patches = x.unfold(dimension=2, size=patch_size, step=patch_size)   # [B, 1, N, P]
    patches = patches.squeeze(1).contiguous()                            # [B, N, P]
    return patches


def masked_patch_mse(pred_patches, target_patches, patch_mask):
    # Use: compute the masked mean squared error.
    # Input: pred_patches: [B, N, P], target_patches: [B, N, P], patch_mask: [B, N], True means masked.
    # Output: loss: scalar, per_sample_masked_mse: [B].
    per_patch_mse = ((pred_patches - target_patches) ** 2).mean(dim=-1)   # [B, N]

    masked_float = patch_mask.float()
    loss = (per_patch_mse * masked_float).sum() / masked_float.sum().clamp_min(1.0)

    per_sample = (per_patch_mse * masked_float).sum(dim=1) / masked_float.sum(dim=1).clamp_min(1.0)
    return loss, per_sample


def compute_masked_metrics(pred_patches, target_patches, patch_mask):
    # Use: compute metrics on masked patches.
    # Input: pred_patches: [B, N, P], target_patches: [B, N, P], patch_mask: [B, N], True means masked.
    # Output: dictionary containing the metrics.
    pred_masked = pred_patches[patch_mask]       # [M, P]
    target_masked = target_patches[patch_mask]   # [M, P]

    pred_flat = pred_masked.reshape(-1)
    target_flat = target_masked.reshape(-1)

    mae = torch.mean(torch.abs(pred_flat - target_flat)).item()
    mse = torch.mean((pred_flat - target_flat) ** 2).item()
    rmse = math.sqrt(mse)

    target_mean = torch.mean(target_flat)
    sse = torch.sum((pred_flat - target_flat) ** 2).item()
    sst = torch.sum((target_flat - target_mean) ** 2).item()
    r2 = 1.0 - sse / (sst + 1e-8)

    pred_center = pred_flat - torch.mean(pred_flat)
    target_center = target_flat - torch.mean(target_flat)
    corr_num = torch.mean(pred_center * target_center)
    corr_den = torch.std(pred_center, unbiased=False) * torch.std(target_center, unbiased=False) + 1e-8
    corr = (corr_num / corr_den).item()

    return {
        "masked_mae": mae,
        "masked_mse": mse,
        "masked_rmse": rmse,
        "masked_r2": r2,
        "masked_corr": corr,
    }


class CNNTransformerMAE(nn.Module):
    # Use: encoder decoder model for training the autoencoder.
    # Input: signal_len: length of the signal, patch_size: size of the patch, stem_dim: dimension of the stem, enc_dim: dimension of the encoder, dec_dim: dimension of the decoder, enc_layers: number of encoder layers, enc_heads: number of encoder heads, dec_layers: number of decoder layers, dec_heads: number of decoder heads, mlp_ratio: ratio of the feedforward dimension to the model dimension, dropout: dropout rate.
    # Output: pure side effect.
    def __init__(
        self,
        signal_len=500,
        patch_size=25,
        stem_dim=64,
        enc_dim=128,
        dec_dim=128,
        enc_layers=4,
        enc_heads=4,
        dec_layers=2,
        dec_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        self.signal_len = signal_len
        self.patch_size = patch_size
        self.num_patches = signal_len // patch_size

        
        # CNN STEM
        # 500 -> (CNN layer 1) -> 100 -> (CNN layer 2) -> 20 time steps (matching the number of patches per signal)
        
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=5, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, stem_dim, kernel_size=5, stride=5, padding=2),
            nn.BatchNorm1d(stem_dim),
            nn.GELU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, signal_len)
            dummy_out = self.cnn_stem(dummy)
            assert dummy_out.shape[-1] == self.num_patches, (
                f"CNN stem output length {dummy_out.shape[-1]} != num_patches {self.num_patches}"
            )

        
        # ENCODER
       
        self.enc_proj = nn.Linear(stem_dim, enc_dim)
        self.enc_pos = nn.Parameter(torch.zeros(1, self.num_patches, enc_dim))

        enc_ff_dim = int(enc_dim * mlp_ratio)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=enc_dim,
            nhead=enc_heads,
            dim_feedforward=enc_ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.enc_norm = nn.LayerNorm(enc_dim)

        
        # DECODER
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, self.num_patches, dec_dim))

        dec_ff_dim = int(dec_dim * mlp_ratio)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=dec_dim,
            nhead=dec_heads,
            dim_feedforward=dec_ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=dec_layers)
        self.dec_norm = nn.LayerNorm(dec_dim)

        # reconstruct raw patch values
        self.patch_head = nn.Linear(dec_dim, patch_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.enc_pos, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, patch_mask):
       
        B, C, T = x.shape
        assert C == 1
        assert T == self.signal_len

        # patch targets for reconstruction loss
        target_patches = patchify_1d(x, patch_size=self.patch_size)   # [B, N, P]

        
        
        stem = self.cnn_stem(x)                    # [B, stem_dim, N]
        tokens = stem.transpose(1, 2)             # [B, N, stem_dim]
        tokens = self.enc_proj(tokens)            # [B, N, enc_dim]
        tokens = tokens + self.enc_pos            # add positional embedding

        # visible tokens only
        visible_mask = ~patch_mask                # [B, N]
        n_visible = visible_mask.sum(dim=1)

        # assumes same mask ratio for every sample -> same number of visible tokens
        assert torch.all(n_visible == n_visible[0]), "All samples in a batch must have same visible count"

        visible_tokens = tokens[visible_mask].view(B, -1, tokens.shape[-1])   # [B, N_visible, enc_dim]

        # encode visible tokens
        enc_visible = self.encoder(visible_tokens)   # [B, N_visible, enc_dim]
        enc_visible = self.enc_norm(enc_visible)

        # simple batch-level representation for monitoring
        sample_repr = enc_visible.mean(dim=1)        # [B, enc_dim]

    
        # DECODER INPUT
        # full length sequence = visible encoded tokens + mask tokens
        dec_visible = self.enc_to_dec(enc_visible)   # [B, N_visible, dec_dim]

        dec_tokens = self.mask_token.repeat(B, self.num_patches, 1)   # [B, N, dec_dim]
        dec_tokens[visible_mask] = dec_visible.reshape(-1, dec_visible.shape[-1])
        dec_tokens = dec_tokens + self.dec_pos

        # decode full sequence
        dec_out = self.decoder(dec_tokens)          # [B, N, dec_dim]
        dec_out = self.dec_norm(dec_out)

        # predict raw patch values
        pred_patches = self.patch_head(dec_out)     # [B, N, patch_size]

        # masked reconstruction loss only
        loss, per_sample_masked_mse = masked_patch_mse(pred_patches, target_patches, patch_mask)

        return {
            "loss": loss,
            "pred_patches": pred_patches,
            "target_patches": target_patches,
            "per_sample_masked_mse": per_sample_masked_mse,
            "sample_repr": sample_repr,
        }
        
def run_pretrain_epoch(model, loader, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    stats = defaultdict(float)
    n_batches = 0
    n_samples_total = 0

    ptb_sum = 0.0
    ptb_count = 0
    ltdb_sum = 0.0
    ltdb_count = 0

    for batch in loader:
        x = batch["x"].to(device)                         # [B, 1, 500]
        patch_mask = batch["patch_mask"].to(device)       # [B, 20]

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            out = model(x, patch_mask)
            loss = out["loss"]

            if train:
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                grad_norm = torch.tensor(0.0)

        # batch-level metrics
        batch_metrics = compute_masked_metrics(
            out["pred_patches"].detach(),
            out["target_patches"].detach(),
            patch_mask.detach(),
        )

        # feature collapse sanity check
        feature_std = out["sample_repr"].detach().std(dim=0, unbiased=False).mean().item()

        B = x.shape[0]
        n_batches += 1
        n_samples_total += B

        stats["loss"] += loss.item() * B
        stats["masked_mae"] += batch_metrics["masked_mae"] * B
        stats["masked_mse"] += batch_metrics["masked_mse"] * B
        stats["masked_rmse"] += batch_metrics["masked_rmse"] * B
        stats["masked_r2"] += batch_metrics["masked_r2"] * B
        stats["masked_corr"] += batch_metrics["masked_corr"] * B
        stats["feature_std"] += feature_std * B
        stats["grad_norm"] += float(grad_norm) * B

        # source-specific loss
        sources = np.array(batch["source"])
        per_sample = out["per_sample_masked_mse"].detach().cpu().numpy()

        ptb_mask = (sources == "PTB")
        ltdb_mask = (sources == "LTDB")

        if ptb_mask.any():
            ptb_sum += per_sample[ptb_mask].sum()
            ptb_count += ptb_mask.sum()

        if ltdb_mask.any():
            ltdb_sum += per_sample[ltdb_mask].sum()
            ltdb_count += ltdb_mask.sum()

    # average metrics
    for k in list(stats.keys()):
        stats[k] /= max(n_samples_total, 1)

    stats["ptb_masked_mse"] = ptb_sum / max(ptb_count, 1)
    stats["ltdb_masked_mse"] = ltdb_sum / max(ltdb_count, 1)

    return dict(stats)


def print_pretrain_stats(epoch, train_stats, val_stats=None):
    msg = (
        f"Epoch {epoch:03d} | "
        f"train_loss={train_stats['loss']:.4f} | "
        f"train_mae={train_stats['masked_mae']:.4f} | "
        f"train_rmse={train_stats['masked_rmse']:.4f} | "
        f"train_r2={train_stats['masked_r2']:.4f} | "
        f"train_corr={train_stats['masked_corr']:.4f} | "
        f"feat_std={train_stats['feature_std']:.4f} | "
        f"PTB_mse={train_stats['ptb_masked_mse']:.4f} | "
        f"LTDB_mse={train_stats['ltdb_masked_mse']:.4f}"
    )

    if val_stats is not None:
        msg += (
            f" || val_loss={val_stats['loss']:.4f} | "
            f"val_mae={val_stats['masked_mae']:.4f} | "
            f"val_rmse={val_stats['masked_rmse']:.4f} | "
            f"val_r2={val_stats['masked_r2']:.4f} | "
            f"val_corr={val_stats['masked_corr']:.4f}"
        )

    print(msg)
