SAMPLING_RATE = 10e6  # sampling rate
DUR_PER_ROW = 1.6e-6  # defines size of the spectrograms
GROUPBY = 160  # for reducing spectrogram size
FINAL_IMAGE_WIDTH = int(DUR_PER_ROW * SAMPLING_RATE)
CATEGORIES = ["P0N#2", "Q3N#3", "Q3N#2", "P0N#1", "nan", "Q3N#1"]
SPECTROGRAM_SIZE = (312, 16)
VALID_SNRS = {"nan", "20.0", "18.0", "16.0"}
PRINT = True
