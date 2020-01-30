# BTS-Crazed

* Solves: 324
* Points: 75
* Category: Forensics

My friend made this cool remix, and it's pretty good, but everyone says there's 
a deeper meaning in the music. To be honest, I can't really tell - the second 
drop's 808s are just too epic.

### Hint 1

https://github.com/JEF1056/riceteacatpanda/raw/master/BTS-Crazed (75)/Save Me.mp3

## Challenge

You were given this mp3 file of what is supposedly a remix (to be honest I have
never looked at the file).

## Solution

As this challenge gave only 75 points and was a `Forensics` challenge the first
instinct is calling `strings` on the file.

```
strings "Save Me.mp3" | grep rtcp
rtcp{j^cks0n_3ats_r1c3}N
```
