import csv
import json
import os
import math
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

def powerspectrum(file):
    ''' Loads the file for manipulation and perform fast fourier transforms, Calculate magnitude and frequency for power spectrum plot.
        :file : The sample file to undergo processing.
    '''
    
    signal, sample_rate = librosa.load(file, sr=22050)
    Fft = np.fft.fft(signal) #fourier transform
    spec = np.abs(Fft)  #magnitude calculation
    f = np.linspace(0, sample_rate, len(spec)) #frequency calculation

    halfspec = spec[:int(len(spec)/2)] #divide magnitude /2
    halffreq = f[:int(len(spec)/2)] #divide freq /2

    plt.figure()
    plt.plot(halffreq, halfspec, alpha=0.4)
    plt.ylabel("Magnitude")
    plt.xlabel("Freq.")
    plt.title("Power spectrum for sample file: " + file)

    plt.show()

def mfccplot(file, n_mfcc):
    ''' Perform short time fourier transforms then plot MFCC representations.
    :file : The file to be represented
    :n_mfcc: The number of MFCCs to be represented in the plot produced by this method.
    '''
    hoplength = 1024 #half of wideband window length
    numfastfourier = 2048 # window in num. of samples. Wideband window length
    signal, sample_rate = librosa.load(file, sr=22050)
    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=numfastfourier, hop_length=hoplength, n_mfcc=n_mfcc)

    plt.figure()
    librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hoplength)
    plt.ylabel("Values of Mel Frequency Ceptstral Coefficients spectrum")
    plt.xlabel("Time")
    plt.colorbar()
    plt.title("MFCC plot for sample file: " + file)
    plt.show()

def processmfccs(path_data, csv_path, num_mfcc=13, numfastfourier=2048, hoplength=1024, totalsegments=5):
    """Performs calculations on dataset of MFCC values and saves with genre code into CSV file
    """
    fieldnames  = ["map", "code", "mfcc"]
    dataset = {"map": [], "code": [], "mfcc": []} #dataset stored as dictionary

    avg_samps_per_segment = int(totalsamps / totalsegments)
    exp_mfcc_perseg = math.ceil(avg_samps_per_segment / hoplength)


    for i, (path, folder, filenames) in enumerate(os.walk(path_data)):
        if path is not path_data:
            nameofgenre = os.path.split(path)[-1] #sorts by genre name
            dataset["map"].append(nameofgenre)
            print("\nCurrently processing genre: " + nameofgenre)
            
            for filename in filenames:
                filepath = os.path.join(path, filename)
                signal, sample_rate = librosa.load(filepath, sr=samplerate)

                for seg in range(totalsegments): #apply transformations to all segments of file
                    first = avg_samps_per_segment * seg
                    current = first + avg_samps_per_segment
                    mfcc = librosa.feature.mfcc(signal[first:current], sample_rate, n_mfcc=num_mfcc, n_fft=numfastfourier, hop_length=hoplength)
                    mfcc = mfcc.T #calculate the mfcc for the segment
                    if len(mfcc) == exp_mfcc_perseg: #store mfcc only with expected number of mfcc per segment
                        dataset["mfcc"].append(mfcc.tolist())
                        dataset["code"].append(i-1)

    # save MFCCs to csv file
    with open(csv_path, "w") as fp: # save MFCCs to csv file
        json.dump(dataset, fp, indent=4)

path_data = 'C:/Users/Adam/Desktop/Project/Data/'
csv_path = "data_10.json"
samplerate = 22050
songlength = 30 # measured in seconds
totalsamps = samplerate * songlength 
        
if __name__ == "__main__":
    processmfccs(path_data, csv_path, totalsegments=10)
    files = ["country.00001.wav", "hiphop.00046.wav", "classical.00005.wav", "rock.00025.wav"]
    for file in files:
        powerspectrum(file)
    for file in files:
        mfccplot(file, 1)
    for file in files:
        mfccplot(file, 14)