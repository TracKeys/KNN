import os
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from PIL import Image


locations = {'A':(12,45), 'B':(34,55), 'C':(11,32),'D':(7,7)}
Oil_Spill_Discovered =[]

# bmp_image = Image.open(os.path.join('./Test','bmpImage.bmp'))
# rgb_image = bmp_image.convert("RGB")

# rgb_image.save('bmpImage.jpg', "JPEG")
# exit()
with open('./Oil_BestEstimator_Model.p', 'rb') as model_file:
    model = pickle.load(model_file)

for ndx,file in enumerate(os.listdir('./Test/New_Images')):
    image = imread(os.path.join('./Test/New_Images/',file))
    image = resize(image,(15, 15,3))
    image =image.flatten() 
    image =np.array([image])
    print(file)
    prediction = model.predict(image)[0]

    if(prediction == 0):
        Oil_Spill_Discovered.append(file)
        location_key, location_value = list(locations.items())[ndx]
        # print(location_value,'Oil Spill' ,file ,   prediction)
        with open("result.txt", "a") as f:
             f.write(f"{location_value} Oil Spill {file} {prediction}\n")

    else:
        print('Normal Image' ,   prediction)
