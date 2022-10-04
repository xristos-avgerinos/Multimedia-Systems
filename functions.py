import numpy as np


# 720x1280 becomes 180x320 containing 3600 4x4 macroblocks.
def upper_level(source, target):
    source = np.array(source)
    target = np.array(target)
    Himg1 = [source]
    Himg2 = [target]
    # with lower resolution ,so we search for the macro-blocks matches
    # We go up to level 3
    for i in range(3):
        source = hierarchical_search(source)  # image
        Himg1.append(source)
        target = hierarchical_search(target)
        Himg2.append(target)
    return Himg1, Himg2


# check every macro-block that had movement in every level

def to_lower_level(movement_blocks,hierimg1,hierimg2):
    search = [8, 16]
    for k in range(len(search)):
        image1 = to_macro_blocks(search[k], hierimg1[1-k])
        image2 = to_macro_blocks(search[k], hierimg2[1-k])
        no_movement_blocks = []
        for i in range(len(movement_blocks)):
            if has_motion(image1[movement_blocks[i]], image2[movement_blocks[i]]):
                continue
            else:
                no_movement_blocks.append(i)
        tmp = []
        for i in movement_blocks:
            if i not in no_movement_blocks:
                tmp.append(i)
        movement_blocks = tmp
    return image1, image2, movement_blocks


# Sub sampling function
def hierarchical_search(image):
    image = np.array(image)
    height, width = image.shape
    hierarchicalImage = []
    # Sub-sampling dividing each time by 2 both height and width
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            # In case of an even number
            try:
                hierarchicalImage.append(image[i][j])
            except:
                continue
    hierarchicalImage = np.array(hierarchicalImage)
    # Reshape the on column array to our sub-sampled dimensions
    hierarchicalImage = np.reshape(hierarchicalImage, (int(height/2), int(width/2)))
    return hierarchicalImage


# motion = 10%  movement or higher
def has_motion(macroblock1, macroblock2):
    diff = np.array(macroblock1 - macroblock2)
    height, width = diff.shape
    res = height*width
    count = res - np.count_nonzero(diff)
    if count > 0.9 * res:
        return False
    else:
        return True


def calculate_sad(macro1, macro2):
    sad = 0
    n = macro1.shape[0]
    for i in range(n):
        for j in range(n):
            sad += abs(int(macro1[i, j]) - int(macro2[i, j]))
    return sad


# Find which of the neighbour macro-blocks has the min sad score
def get_sad(mc1, mc2, i):
    block = []
    diff = [calculate_sad(mc2[i], mc1[i])]
    block.append(i)
    width = mc1.shape[1]

    # Right macro-block
    if i + 1 <= width:
        diff.append(calculate_sad(mc2[i], mc1[i + 1]))
        block.append(i + 1)

    # Left macro-block
    if i - 1 >= 0:
        diff.append(calculate_sad(mc2[i], mc1[i - 1]))
        block.append(i - 1)

    # Below macro-block
    if i + width <= width**2:
        diff.append(calculate_sad(mc2[i], mc1[i + width]))
        block.append(i + width)

    # Above macro-block
    if i - width >= 0:
        diff.append(calculate_sad(mc2[i], mc1[i - width]))
        block.append(i - width)

    # Diagonal below right
    if i + width + 1 <= width**2:
        diff.append(calculate_sad(mc2[i], mc1[i + width + 1]))
        block.append(i + width)

    # Diagonal below left
    if i + width - 1 <= width**2:
        diff.append(calculate_sad(mc2[i], mc1[i + width - 1]))
        block.append(i + width)

    # Diagonal above right
    if i - width + 1 >= 0:
        diff.append(calculate_sad(mc2[i], mc1[i - width + 1]))
        block.append(i - width)

    # Diagonal above left
    if i - width - 1 >= 0:
        diff.append(calculate_sad(mc2[i], mc1[i - width - 1]))
        block.append(i - width)

    # Get the index of the neighbour block with the min sad score
    return block[diff.index(min(diff))]


# MacroBlocks to Image
def create_image(height, width, macroblocks):
    c = 1
    for i in range(height):
        tmp = np.array(macroblocks[i * width])
        for j in range(width-1):
            tmp = np.concatenate((tmp, macroblocks[c]), axis=1)
            c += 1
        c += 1
        # In the first iteration set the output as the rebuilt row
        if i == 0:
            output = tmp
        # Else add the new row to the half-rebuilt image
        else:
            output = np.concatenate((output, tmp), axis=0)
    return output


# Split the image into macro-blocks
def to_macro_blocks(k, arr):
    macro_blocks = []
    for i in range(0, arr.shape[0] - k+1, k):
        for j in range(0, arr.shape[1] - k+1, k):
            macro_blocks.append(arr[i:i+k, j:j+k].astype('int32'))
    macro_blocks = np.array(macro_blocks)
    return macro_blocks

