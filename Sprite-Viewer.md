# Sprite Viewer

* Category: Web

* Points: 400

* Solves: 37


# Description

The developer of that work in progress game thinks there may
have been a bug on his end, and has asked us to get the md5
hash of the goblin sprite image.

Here's the place where he got the images... but... they're
stuck in a web player!!! How will you get the hash out of that?!

Hint1: https://riceteacatpanda.wtf/spview 

Hint2:Insert flag in MD5 format, without the rtcp{}:
e7ad2f601c93ea0dab82416581db2dfd

# Challenge

Looks like we got to figure out how to extract a sprite from a Unity web game.


# Solution

The challenge loads up a Unity Web Player. It has a number of characters you can swap
through. There are options to view different sprite animations. Looks like a character
creator for a game. But we want just want the goblin sprite.

First thing to do is open up the source code of the page. There you can find "Builds.js" 
Go to it's URL and you get the file. Open it and you find references to a number of other 
files that Unity will load up. We are interested in the "Builds.data.unityweb" This file 
should contain all the assets used. Extracting them can be a bit tricky. The file is 
gzipped and can be unzipped. This gets you some plain text but not an easy way to get the 
files.

Fortunately, there is a tool called DevXUnityUnpackerTools. This allows us to browse the
contents of the file. Little looking and we find "goblin.png" Select it and we can "save
as a PNG" Hit that the is extracted and saved. 

Now we just need to hash the image with MD5. To do that I uploaded it to Cyber Chef and 
set the recipe to "MD5" Bake and we are done.
