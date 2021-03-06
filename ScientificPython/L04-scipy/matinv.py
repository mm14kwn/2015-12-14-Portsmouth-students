#!/usr/bin/env python

import numpy as np
from scipy import linalg

A=np.array( [[1,3,5],[2,5,1],[2,3,8]] )

invA=linalg.inv(A)
print invA

#check inverse gives identity
identity=A.dot(linalg.inv(A))

#round each element to zero dp
print np.abs(np.around(identity,0))

