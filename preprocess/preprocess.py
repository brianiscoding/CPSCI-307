import os
import h5py
import numpy as np
import pandas as pd
import constants as c


def create_zip_signals_labels_snrs(id_group, id_subset):
    file_signals = f"SimulatedRadarWaveforms/Group{id_group}/group{str(id_group)}_subset_{str(id_subset)}.mat"
    file_features = f"SimulatedRadarWaveforms/Group{id_group}/group{id_group}_subset_CSVInfo/group{str(id_group)}_waveformTableSubset_{str(id_subset)}.csv"
    if not os.path.isfile(file_signals) or not os.path.isfile(file_features):
        return

    h5pyObj = h5py.File(file_signals, "r")
    var_waveform = f"group{str(id_group)}_waveformSubset_{str(id_subset)}"
    signals = h5pyObj[var_waveform][()].view(complex)

    df = pd.read_csv(file_features)
    labels = df["BinNo"].tolist()
    snrs = df["SNR"].tolist()

    return zip(signals, labels, snrs)


def conditional_print(string):
    if not c.PRINT:
        return
    print(string)


# This method creates PSDs and spectrograms from I,Q samples
# The PSD values along with ground truth are written ina file
def subset_to_csvs(id_group, id_subset):
    zip_signals_labels_snrs = create_zip_signals_labels_snrs(id_group, id_subset)
    if not zip_signals_labels_snrs:
        conditional_print(f"group {id_group} subset {id_subset}: failed")
        return

    spectrogram_count = 0
    for i, (signal, label, snr) in enumerate(zip_signals_labels_snrs):
        # filter (rec by karyn doke)
        if str(snr) not in c.VALID_SNRS:
            continue

        spectrogram = iq_to_spectrogram(signal)
        df = transform_spectrogram(spectrogram)
        file = f"data/{str(label)}/{id_group}_{id_subset}_{i}.csv"
        df.to_csv(file, index=False, header=False)
        spectrogram_count += 1

    conditional_print(f"group {id_group} subset {id_subset}: {spectrogram_count}/200")


def iq_to_spectrogram(iq_data):
    in_data = iq_data.reshape(-1, c.FINAL_IMAGE_WIDTH)
    S = np.fft.fftshift(
        np.fft.fft(in_data, c.FINAL_IMAGE_WIDTH, axis=1) / c.FINAL_IMAGE_WIDTH, axes=1
    )
    S = 20 * np.log10(np.abs(S))
    L = S.shape[0] / c.GROUPBY
    S = np.reshape(
        S[0 : int(L) * c.GROUPBY, :], (int(L), c.GROUPBY, c.FINAL_IMAGE_WIDTH)
    )
    S = np.amax(S, axis=1)
    return S


# 1.2mb > 179kb
def transform_spectrogram(spectrogram):
    psd = (spectrogram.reshape((1, -1))).ravel().tolist()
    array = np.array(psd)
    array_reshaped = array.reshape(c.SPECTROGRAM_SIZE)
    # normalize from 0 to 100
    array_resized = (
        (array_reshaped - np.min(array_reshaped))
        / (np.max(array_reshaped) - np.min(array_reshaped))
        * 100
    )
    # save space
    array_int = array_resized.astype(int)
    df = pd.DataFrame(array_int)
    return df


def create_directory_categories():
    root_data = "data"
    if not os.path.exists(root_data):
        os.mkdir(root_data)
    for category in c.CATEGORIES:
        directory_category = f"{root_data}/{category}"
        if not os.path.exists(directory_category):
            os.mkdir(directory_category)


def main():
    create_directory_categories()
    id_group = 4
    # ids_subset = range(1, 26)
    ids_subset = range(26, 51)
    for id_subset in ids_subset:
        subset_to_csvs(id_group, id_subset)


if __name__ == "__main__":
    main()
