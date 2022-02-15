# from DepthEstimation.inference import main
# main()
# print('Depth Estimation Done!')

import os
from Inpainting import main
print('impainting started')
for i in os.listdir("Input"):
  main.inpaint(i)
