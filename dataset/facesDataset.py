import os
import pandas as pd
import numpy as np
import imageio
from PIL import Image
import os
import torchvision.transforms as Tfs

DATASET_PATH ="dataset/faces/lfw-deepfunneled/lfw-deepfunneled/"
ATTRIBUTES_PATH = "dataset/faces/info.txt"

def fetch_dataset(dx=80,dy=80, dimx=45,dimy=45):
  
  df_attrs = pd.read_csv(ATTRIBUTES_PATH, sep='\t', skiprows=1,) 
  df_attrs = pd.DataFrame(df_attrs.iloc[:,:-1].values, columns = df_attrs.columns[1:])
  
  photo_ids = []
  for dirpath, dirnames, filenames in os.walk(DATASET_PATH):
      for fname in filenames:
          if fname.endswith(".jpg"):
              fpath = os.path.join(dirpath,fname)
              photo_id = fname[:-4].replace('_',' ').split()
              person_id = ' '.join(photo_id[:-1])
              photo_number = int(photo_id[-1])
              photo_ids.append({'person':person_id,'imagenum':photo_number,'photo_path':fpath})

  photo_ids = pd.DataFrame(photo_ids)
  df = pd.merge(df_attrs,photo_ids,on=('person','imagenum'))

  assert len(df)==len(df_attrs),"lost some data when merging dataframes"
  
  all_photos = df['photo_path'].apply(imageio.imread)\
                              .apply(lambda img:img[dy:-dy,dx:-dx])\
                              .apply(lambda img: np.array(Image.fromarray(img).resize([dimx,dimy])))

  all_photos = np.stack(all_photos.values).astype('uint8')
  all_attrs = df.drop(["photo_path","person","imagenum"],axis=1)
  
  return all_photos,all_attrs

data, attrs = fetch_dataset()

class DataSet():
    def __init__(self, data=data):
        self.data = data
        self.to_tensor = Tfs.ToTensor()
    def __getitem__(self, i):
        img = self.to_tensor(self.data[i])
        return img, 0
    def __len__(self):
        return self.data.shape[0]
    