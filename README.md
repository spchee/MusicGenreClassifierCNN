# Music-Genre-Classifier-CNN

This program will determine the genre of a given song/songs out of the 5 following:

Metal, Rock, Hip-Hop, EDM, and Pop.

![alt text](https://i.imgur.com/TNI9VmU.png)

When predicting individual snippets, it had an accuracy of about 52%, however when predicting full songs, this increased quite a bit.

I used the Google AudioSet during training - https://research.google.com/audioset/dataset/index.html

And you can find the model I developed here - https://mega.nz/file/oA1WDKjL#_hBK60YVySvi-DKwqnFkS12TrRm8qBD7vpFj3YgNSwA

There's a basic GUI using PyQT5 which will change the ID3 tag of the songs you predict, however this is extremely buggy, alternatively theres the CLI which isn't as buggy, but I didn't bother implementing it changing ID3 tags or anything like that.

There's also a redundant password menu in the GUI, this is just because it was used in a school project and required it. The password is nothing.

This is my first large project using CNNs and Keras, so the code is a bit all over the place.

