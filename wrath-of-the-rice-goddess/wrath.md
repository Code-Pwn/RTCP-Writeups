# Wrath of the Rice Goddess

* Solves: 47
* Points: 4000
* Category: Rice Goddess

So, uh, you see, when you wetuwnyed my uwu back to me, the wice goddess got a 
bit angwy - nyow she has a giant panda weady to sit on my uwu stowage... can you 
tawk to it? >w<

Quest: Talk to the giant panda on discord

#### Hint 1

Data linearization and generalization tend to make a lot of things lead to Rome.

... or I think that's how the saying goes

#### Hint 2

Python is more or less required for this. Try making a byte encoder!

#### Hint 3

I would try sweet-talking the panda, maybe start with something like 
`I love rice, tea, cats, and pandas`

#### Hint 4

Never, **EVER** trust the developer. (but do trust this hint)

## Challenge

You were given two files.

dontbeconcerned.whatsoever:
```
=0GJUhKgKiIjSihnGY_g_o-dn-P2HiJ_wasilW_qUlL3
```

pandaspeak_encoder.py:
```python
import cryptography
from cryptography.fernet import Fernet
import random

class BytesIntEncoder:
    @staticmethod
    def encode(b: bytes) -> int:
        return int.from_bytes(b, byteorder='big')

    @staticmethod
    def decode(i: int) -> bytes:
        return i.to_bytes(((i.bit_length() + 7) // 8), byteorder='big')

file = open('key.key', 'rb')
key = file.read()
file.close()

while True:
    response = input("Paste your message here: ")
    f = Fernet(key)
    encoded0=f.encrypt(response.encode())
    encoded=str(BytesIntEncoder.encode(encoded0))

    #generate how many splits we want to have
    numofsplits = random.randint(int(len(encoded)/5),int(len(encoded)/2))
    loc = []
    for i in range(0, numofsplits):
        loc.append(random.randint(2, len(encoded)))

    response = ""
    for char in encoded:
        response = response + chr(int(char)+97)

    for inst in loc:
        response = response[:inst] + '|' + response[inst:]

    buffer = response.split("|")
    response = ""
    for segment in buffer:
        noisetype = random.randint(1,5)
        if noisetype ==1:
            segment = "ra" + segment + "wr "
        elif noisetype == 2:
            segment = "gr" + segment + "rr "
        elif noisetype == 3:
            segment = "sq" + segment + "ak "
        elif noisetype == 4:
            segment = "hu" + segment + "ff "
        elif noisetype == 5:
            segment = "ch" + segment + "rp "
        response = response + segment
    print()
    print(response)
    print()
```

The `pandaspeak_encoder.py` simply encrypts the given message with a symmetric
encryption algorithm called `Fernet` and modifies the cipher text a bit by 
splitting it into a random amount of elements with a random length. These bits
then get some random noise added at the front and back and this is then printed.

## Solution

The first goal is to get the actual cipher text from the mess that is "printed"
out. For that we follow the script from the bottom up and undo everything that
was done to the cipher text. To remove the "noise" you just take each block 
and remove the first and last two characters. Then you remove the spaces. After
that is done you simply turn each byte into number again and cast the entire
string into an integer. This int is now your encoded cipher text. Luckily the 
way to decode the message was also given in the script.

```python
class BytesIntEncoder:
    @staticmethod
    def decode(i: int) -> bytes:
        return i.to_bytes(((i.bit_length() + 7) // 8), byteorder='big')

m = input("Message: ")
m_orig = "".join([x[2:-2] for x in m.split()])
m_orig = int("".join([str(ord(x)-97) for x in m_orig]))
cipher = BytesIntEncoder.decode(m_orig)
```

To decrypt the cipher text we have to find out the key that was used. Here hint
4 points us into the right direction. If we don't trust the developer then we 
should indeed be concerned with the file `dontbeconcerned.whatsoever`. The 
content of that file looks somewhat like base64 with some added characters. 
Since it is not actually base64 we just reversed the string and tried that as 
the key.

```python
from cryptography.fernet import Fernet

f = Fernet("=0GJUhKgKiIjSihnGY_g_o-dn-P2HiJ_wasilW_qUlL3"[::-1])
```

Now to getting an actual encrypted message. Hint 3 tells us to tell the panda 
that `I love rice, tea, cats, and pandas`. In the discord server of the 
organizers was a bot called `Giant Panda` and if you send that bot this message,
its response will look something like
```
hudjdff grrr huaff huggff rajgwr grjifdrr grrr sqbak grijdbgfrr chhafrp 
grjjgerr sqak rafbcwr sqak huff chgibifcrp rabwr chchacrp sqak hujff chfrp huff 
grgcjcrr hudcfff grcggrr huachhff chhrp rabiwr huff rawr grhfrr raaeaiwr chjrp 
sqghak chaairp chjjbdhcdrp chfdrp huiff grjdcrr chrp grjhdrr grerr huff sqak 
huejheff sqceeeaak chefrp charp sqfeeak grcrr huff raffdabiwr sqeak sqhhahak 
sqbbibajgjjbeegfak chgdegjjdarp huaiedcgjfbifjff huaff huff huff hugjiff 
hueihff chfhrp rawr sqgak rajdjhebcwr rajwr chjgirp grrr hucfff grjrr rawr huff 
grrr sqaeifcegbhadak huff huff sqgebhbjgfak huff rahbcjwr huaiff sqak chrp chrp 
rahabawr chhgjfrp sqak hufjff huceff chaddfbhrp racefeiwr sqjacak hubaicadgcff 
hugfjfhfajbdcjafbhgff sqjefak hucff chjrp chacarp raahjcfhcwr 
hubibjcdebjjfddaff chahgfbgbgicjrp
```

Note that every block starts and ends with one of the types of "noise".

If you now decrypt that message you get the following text:
```
especially pandas
```

Now to actually get the flag you have to continue the conversation by sending 
the panda exactly what it send you just in plain english.

```
I love rice, tea, cats, and pandas
especially pandas
yeah do you want to tell me how much you love pandas?
i love them so much, i'd cuddle them all day
so? the rice goddess does that too.
```

And after that heart warming conversations about pandas the last encrypted 
message the bot will send you looks decrypted like this:
```
darn!!! here's your flag: rtcp{pand4z_just_w4nt_cudd13z_fr0m_y0u!}
```

## Script

```python
from cryptography.fernet import Fernet

class BytesIntEncoder:
    @staticmethod
    def decode(i: int) -> bytes:
        return i.to_bytes(((i.bit_length() + 7) // 8), byteorder='big')

f = Fernet("=0GJUhKgKiIjSihnGY_g_o-dn-P2HiJ_wasilW_qUlL3"[::-1])

while True:
    m = input("Message: ")
    m_orig = "".join([x[2:-2] for x in m.split()])
    m_orig = int("".join([str(ord(x)-97) for x in m_orig]))
    print()
    print(f"Decoded: {BytesIntEncoder.decode(m_orig)}")
    print(f"Message: {f.decrypt(BytesIntEncoder.decode(m_orig)).decode()}")

```