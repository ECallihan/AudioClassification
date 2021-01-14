'''
This file provides the methods to download and extract features from the
UrbanSound8K dataset.
'''
import urllib.request
import tarfile
import numpy as np
import pandas as pd
import librosa
import csv
import os

us8k_dl_location = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
tar_dl_name = "urbansound.tar.gz"
np_save = "raw_data.npy"
csv_save = 'UrbanSound8K/dataset.csv'
data_directory = 'UrbanSound8K/audio'
metadata_loc = "drive/MyDrive/UrbanSound8K/metadata/UrbanSound8K.csv"


def dl_extract_tar(tarLocation=us8k_dl_location, newTarName=tar_dl_name):
    """
    This method downloads an opensource copy of the UrbanSound8K dataset in the .tar format
    and extracts it into the working directory.
    :param tarLocation: Web address of the download link for the data.
    :param newTarName: This will be the name of the .tar file downloaded.
    :return: None.
    """
    urllib.request.urlretrieve(tarLocation, newTarName)
    tar = tarfile.open(newTarName)
    tar.extractall()
    tar.close()
    print(f'finished with download and extraction of {tarLocation}.')


def get_spectral_values(saveFileName=csv_save, audioDirectory=data_directory):
    """
    This method creates a CSV file holding the file name, 26 total spectral values for each audio
    clip and corresponding labels for each clip.
    :param saveFileName: This is the CSV file where all data is stored.
    :param audioDirectory: The path to the UrbanSound8K/audio folder.
    :return: None. A CSV file is created for the data in the working directory.
    """
    us8k = 'air_conditioner,car_horn,children_playing,dog_bark,drilling,' \
           'engine_idling,gun_shot,jackhammer,siren,street_music'.split(sep=",")

    # Create a header for the CSV file
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    print(header)

    # Save Spectral feature values to a CSV file
    on_file = 0
    file = open(saveFileName, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    for i in range(1, 11):
        for filename in os.listdir(f'{audioDirectory}/fold{i}'):
            clip = f'{audioDirectory}/fold{i}/{filename}'
            if clip[-3:] == "wav":
                on_file = on_file + 1
                print(f'On File: {on_file}')
                y, sr = librosa.load(clip, mono=True)
                rms = librosa.feature.rms(y=y)
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                to_append += f' {us8k[int(filename.split(sep="-")[1])]}'
                file = open(saveFileName, 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())


def get_melspec_data(csv_path=metadata_loc, audio_path=data_directory, save_path=np_save):
    """
    This method produces mel-spectrogram data from the audio files in the UrbanSound8K dataset.
    It requires having the US8K dataset available either locally or remotely such as in Google Drive.
    To account for dynamic length audio, the frequency values from the resultant mel-spectrogram images
    are averaged across their time dimension. This gives a feature array with 128 values.
    :param audio_path: filepath to UrbanSound8K/audio
    :param save_path: filepath to save the processed arrays of feature and label values.
    :param csv_path: filepath to the UrbanSound8K/metadata/UrbanSound8K.csv file in the US8K
    data folder.
    :return: An array of the features for each audio features derived from the audio files and their
    corresponding labels is saved to the path from the save_path parameter. This can then be used
    for Convolutional Neural Network processing.
    """
    df = pd.read_csv(csv_path)
    feature = []
    label = []
    for i in range(8732):
        file_name = f'{audio_path}/fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast', )
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, ).T, axis=0)
        feature.append(mels)
        label.append(df["classID"][i])
    temp = [feature, label]
    temp = np.array(temp)
    data = temp.transpose()
    np.save(save_path, data, allow_pickle=True)


def run():
    """
    This function Runs the entire download, extraction and data formatting process.
    :return: None.
    """
    dl_extract_tar()
    get_melspec_data()
    get_spectral_values()
    print("Preprocessing complete")


if __name__ == '__main__':
    '''
    This sequence of commands will download, extract, and process the dataset for use in 
    machine learning algorithms.
    '''
    run()
