# extremely weak feedback training for the 3 local experts.

import torch

'''
1. determine 3 training sets, one unlabeded training set, a validation set (sperated into 3 subvalidation sets) and an expert (One hour)
2. pre-train the 3 networks on 3 subfield domains training set. To see the Dice loss or pixel-wised loss for 3 local experts in different validation sets. (four hours with the reuse of )

3. train the 3 local experts with the help of the expert with extremely weak feadbacks. 
4. validate that the three local experts perform well on all 3 validation set and their ensemble results are near the fully supervised one. 


'''