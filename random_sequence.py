import numpy as np
import sys

length = int(sys.argv[1])

sequence = ''.join(np.random.choice(['A', 'C', 'G', 'T'], size=length).tolist())
print(sequence)
