#%%
import cv2

#%%
def dhash(image, hashSize=8):
    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash and return it
    return sum([2**i for (i, v) in enumerate(diff.flatten()) if v])


# %%
red_chair_hash = dhash(
    cv2.imread(
        "/home/sicara/R&D/few-shot-style-transfer/src/style_transfer/examples/red_chair.png"
    )
)
red_chair_hash_2 = dhash(
    cv2.imread(
        "/home/sicara/R&D/few-shot-style-transfer/src/style_transfer/examples/red_chair.png"
    )
)
content1_hash = dhash(
    cv2.imread(
        "/home/sicara/R&D/few-shot-style-transfer/src/style_transfer/examples/content1.png"
    )
)
content1style1_hash = dhash(
    cv2.imread(
        "/home/sicara/R&D/few-shot-style-transfer/src/style_transfer/examples/content1-style1.png"
    )
)

print(red_chair_hash == red_chair_hash_2)
print(red_chair_hash_2 == content1_hash)
print(content1_hash == content1style1_hash)
# %%
