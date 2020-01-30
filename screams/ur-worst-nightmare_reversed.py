import cv2
import soundfile
import numpy
import random

# read in the soundfile
sound_file = soundfile.SoundFile("oh-gawd-plsno.wav", 'r')
sound = sound_file.read()
sound_file.close()

# get the image with the flag
image = cv2.imread("pls-no.jpg")
height = image.shape[0]

# initialize the wav_data with 2 channels and all zero values
wav_data = numpy.zeros((height*height, 2), dtype=numpy.float64)

i = 0
# iterate over pixels from image
for h in range(0, height):
    for w in range(0, height):
        # get rgb value of the current pixel
        r, g, b = image[h][w][0], image[h][w][1], image[h][w][2]
        # convert each value to string and pad with zeroes
        r, g, b = str(r).zfill(3), str(g).zfill(3), str(b).zfill(3)
        # sum the values up
        result = "0.00" + r + g + b
        # randomize the sign for every value
        if random.randint(0,1) == 1:
            result = "-" + result
        # write stuff from the soundfile to channel1 and the float from the rgb to channel2
        wav_data[i] = (sound[i][0]*2, float(result))
        i += 1

# write the resulting data to the output wav file
soundfile.write('out.wav', wav_data, 44100, 'FLOAT')