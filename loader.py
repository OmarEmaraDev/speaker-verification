import os
import numpy
from random import Random
from scipy.io import wavfile
from dataclasses import dataclass
from python_speech_features import mfcc
from dataset import metadata as datasetMetadata

@dataclass
class SpeakerFile:
    samplingRate: int
    samples: numpy.ndarray
    index: int

def getSpeakerNames():
    return list(datasetMetadata.metadata.keys())

def getSpeakerFilePaths():
    dataDirectoryPath = os.path.join(os.path.dirname(datasetMetadata.__file__), "recordings")
    fileNames = os.listdir(dataDirectoryPath)
    return [os.path.join(dataDirectoryPath, name) for name in fileNames]

def readSpeakerFiles():
    speakerFilePaths = getSpeakerFilePaths()
    speakerNames = getSpeakerNames()
    getIndex = lambda path: speakerNames.index(path.split("_")[-2])
    return [SpeakerFile(*wavfile.read(path), getIndex(path)) for path in speakerFilePaths]

def readCleanedUpSpeakerFiles():
    speakerFiles = readSpeakerFiles()
    maximumSamples = max(speakerFile.samples.shape for speakerFile in speakerFiles)
    for speakerFile in speakerFiles:
        speakerFile.samples.resize(maximumSamples)
    return speakerFiles

def loadSpeakerFiles():
    speakerFiles = readCleanedUpSpeakerFiles()
    Random(0).shuffle(speakerFiles)
    return speakerFiles

def loadDataSet(speakerIndex):
    speakerFiles = loadSpeakerFiles()
    images = numpy.stack([mfcc(f.samples, f.samplingRate) for f in speakerFiles])
    labelsGenerator = (f.index == speakerIndex for f in speakerFiles)
    labels = numpy.fromiter(labelsGenerator, dtype = numpy.uint8)
    return images, labels
