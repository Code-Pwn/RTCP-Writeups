# I Love You 3000
- Points: 700
- Category: Cryptography

❤️ 144, 588, 1869, 1425, 1267, 1708, 1588, 1600, 1889, 1497, 482, 696, 731, 337, 491, 1314, 437, 1514, 1384, 1561, 419, 382, 835, 325, 1835, 1562, 1092 💔

### Hints:
- I don't read books... do/should you?
- submit in the form rtcp{OHMYGAWDTHISISAWHOLEWORD}

## Challenge
The challenge gives us a bunch of numbers and it doesn't take long to figure out it's a book cipher. Book cipher requires a key(s) and a book/novel/article. We have the keys however finding the book/novel/article was the real challenge. I tried decoding the cipher with the whole and various other snippets of the Avengers:End Game movie script (I Love You 3000 is a line mentioned in the movie) but to no avail :(

Reading the title over and over again led me to look at it in a differnt perspective (As you should for CTFs) and figured maybe the challenge is actually loosely hinting to ILoveYou virus, which made more sense. Thinking the "book" in this case is probably the source code for the virus I did some quick googlefu and got what I needed:
<https://raw.githubusercontent.com/onx/ILOVEYOU/master/LOVE-LETTER-FOR-YOU.TXT.vbs>

Copying the keys and source code, we dump what we have in <https://www.dcode.fr/book-cipher> and we get the following output `RTCPI10V3H0WBROKENMY3MAILZR`
Making sure to follow the flag format and taking note of the hint provided we have our flag: `rtcp{I10V3H0WBROKENMY3MAILZR}`
