# Fingerprint

`Fingerprint` is an experimental audio tool for making audio summaries of large collections of music. I made it as part of the [Transition EP](www.robclouth.com/transition) to summarise my entire musical output since 2003. It compresses weeks-worth of audio into minutes using an algorithm akin to granular synthesis but over much larger timescales.

It was tested on OSX, but should work on other systems too.
It requires python 3.6.

To use it first download the repo then install the dependencies:

`> pip install librosa numpy pydub`

Then `cd` to the downloaded repo and run: 

`python main.py -h` 

for help on the various params. 

If you want to use the default parameters, run: 

`python main.py "path_to_your_music_folder"` 

and see what you get. Because the algorithm randomly fluctuates the grain sizes and spacing, it's hard to accurately calculate the total length of the summary. The default settings compressed all of my music since 2003 down to 88 minutes. Just tweak and see what you get. It all sounds pretty interesting.

P.S. Ignore the commit history of this project. It's ugly and I couldn't be bothered to delete it.





