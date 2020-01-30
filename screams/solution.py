import soundfile
import math
from PIL import Image

## helper function to retrieve color values
def get_color(i):
    result = 0
    if i != '':
        result = int(i)
    return result

## read in soundfile's datapoints and samplerate
data, samplerate = soundfile.read("./aaaaaaaaaaaaaaaaaa.wav")

## calculate flag image's height
height = int(math.sqrt(int(len(data))))

## create a new Image object for the flag image
image = Image.new("RGB", (height, height))
px = image.load()

## iterate over every pixel from the newly created image object
j = 0
for h in range(height):
    for w in range(height):
        ## get both channels from datapoint
        # ch1: values from original sound file (boring)
        # ch2: rgb values from jpg (interesting)
        _, ch2 = data[j]

        ## convert value to string
        rgb = str(ch2)

        ## remove sign
        if rgb[0] == "-":
            rgb = rgb[1:]

        ## remove "0.00" at the beginning
        rgb = rgb[4:]

        ## get r,g,b
        r = get_color(rgb[0:3])
        g = get_color(rgb[3:6])
        b = get_color(rgb[6:9])

        ## set img r,g,b
        # px[h, w] = (r, g, b)
        px[height-h-1, w] = (b, g, r) # result is nicer then the more intuitive line above

        j += 1

## rotate and save resulting image
rotated = image.rotate(90)
rotated.save("result.jpg")