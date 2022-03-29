import os
import sys
from time import time, sleep
from itertools import permutations
import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.tree import DecisionTreeClassifier
    
from PIL import Image, ImageDraw, ImageFont
from torch import positive


def pil2cv(imgPIL):
    imgCV_RGB = np.array(imgPIL, dtype = np.uint8)
    imgCV_BGR = np.array(imgPIL)[:, :, ::-1]
    return imgCV_BGR

def cv2pil(imgCV):
    imgCV_RGB = imgCV[:, :, ::-1]
    imgPIL = Image.fromarray(imgCV_RGB)
    return imgPIL
    

def calc_deg(df, n1, n2, n3):

  vec_a = df[[n1+'x', n1+'y']].values - df[[n2+'x', n2+'y']].values
  vec_b = df[[n2+'x', n2+'y']].values - df[[n3+'x', n3+'y']].values

  degs = []
  for a, b in zip(vec_a, vec_b):
    length_vec_a = np.linalg.norm(a)
    length_vec_b = np.linalg.norm(b)
    inner_product = np.inner(a, b)

    cos = inner_product / (length_vec_a * length_vec_b)
    rad = np.arccos(cos)
    deg = np.rad2deg(rad)
    degs.append(deg)

  return np.radians(np.array(degs))

def preprocessing(df, n1, n2, n3, n4, feature=[]):
  out = pd.DataFrame()

  # 角度補正  
  rad = np.arctan2(df['9y'], df['9x'])
  if not feature or 'rad' in feature: out['rad'] = rad
  r = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])[:,:,0]
    
  for j in range(21):
    df[[str(j)+'x', str(j)+'y']] = df[[str(j)+'x', str(j)+'y']] @ r

  # 極座標 r, θ
  x = df[[n1+'x', n2+'x', n3+'x', n4+'x']].values
  y = df[[n1+'y', n2+'y', n3+'y', n4+'y']].values
  x = np.cumsum(x, axis=1)
  y = np.cumsum(y, axis=1)

  r = np.sqrt(x**2+y**2)
  theta = np.arctan2(y, x)

  if not feature or 'theta1' in feature: out['theta1'] = theta[:, 1] - theta[:, 0]
  if not feature or 'theta2' in feature: out['theta2'] = theta[:, 2] - theta[:, 1]
  if not feature or 'theta3' in feature: out['theta3'] = theta[:, 3] - theta[:, 2]
  
  if not feature or 'r1' in feature: out['r1'] = r[:, 1] - r[:, 0]
  if not feature or 'r2' in feature: out['r2'] = r[:, 2] - r[:, 1]
  if not feature or 'r3' in feature: out['r3'] = r[:, 3] - r[:, 2]

  for p in permutations([n1, n2, n3, n4], 3):
    if not feature or 'a'+''.join(p) in feature: out['a'+''.join(p)] = calc_deg(df, p[0], p[1], p[2])
  
  # 2点間の角度  
  if not feature or 'd'+n1 in feature: out['d'+n1] = np.degrees(np.arctan2(df[n1+'y'], df[n1+'x']))
  if not feature or 'd'+n2 in feature: out['d'+n2] = np.degrees(np.arctan2(df[n2+'y']-df[n1+'y'], df[n2+'y']-df[n1+'x']))
  if not feature or 'd'+n3 in feature: out['d'+n3] = np.degrees(np.arctan2(df[n3+'y']-df[n2+'y'], df[n3+'y']-df[n2+'x']))
  if not feature or 'd'+n4 in feature: out['d'+n4] = np.degrees(np.arctan2(df[n4+'y']-df[n3+'y'], df[n4+'y']-df[n3+'x']))

  if not feature or 'd'+n1+n3 in feature: out['d'+n1+n3] = np.degrees(np.arctan2(df[n3+'y']-df[n1+'y'], df[n3+'y']-df[n1+'x']))
  if not feature or 'd'+n2+n4 in feature: out['d'+n2+n4] = np.degrees(np.arctan2(df[n4+'y']-df[n2+'y'], df[n4+'y']-df[n2+'x']))

  if not feature or 'd'+n1+n4 in feature: out['d'+n1+n4] = np.degrees(np.arctan2(df[n4+'y']-df[n1+'y'], df[n4+'y']-df[n1+'x']))
  
  # under is n4
  if not feature or 'under is '+n4 in feature: out['under is '+n4] = (np.argmin(df[[n1+'y', n2+'y', n3+'y', n4+'y']].values, axis=1)) == 3

  # top is n4
  if not feature or 'top is '+n4 in feature: out['top is '+n4] = (np.argmax(df[[n1+'y', n2+'y', n3+'y', n4+'y']].values, axis=1)) == 3

  # n1 vs n3
  if not feature or n1+' vs '+n3 in feature: out[n1 + ' vs ' + n3] = df[n1+'y'] < df[n3+'y']

  # dist 0 n1
  if not feature or '0_'+n1 in feature: out[f'0_{n1}'] = np.sqrt((df['0x']-df[n1+'x'])**2 + (df['0y']-df[n1+'y'])**2 + 1e-10)

  # dist 0 n2
  if not feature or '0_'+n2 in feature: out[f'0_{n2}'] = np.sqrt((df['0x']-df[n2+'x'])**2 + (df['0y']-df[n2+'y'])**2 + 1e-10)

  # dist 0 n3
  if not feature or '0_'+n3 in feature: out[f'0_{n3}'] = np.sqrt((df['0x']-df[n3+'x'])**2 + (df['0y']-df[n3+'y'])**2 + 1e-10)

  # dist 0 n4
  if not feature or '0_'+n4 in feature: out[f'0_{n4}'] = np.sqrt((df['0x']-df[n4+'x'])**2 + (df['0y']-df[n4+'y'])**2 + 1e-10)

  return out

import joblib

class BackEnd(object):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic

        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.mp_drawing = mp.solutions.drawing_utils

        self.a_dtree = joblib.load(os.path.join('weight', 'a.tree'))
        self.b_dtree = joblib.load(os.path.join('weight', 'b.tree'))
        self.c_dtree = joblib.load(os.path.join('weight', 'c.tree'))

    def detection(self, pos):
        start = time()
        df = self.norm_mp_pos(pos)

        features = ['r3', 'a171819', 'a171920', 'a182017', 'a182019', 'a201918', 'd1720']
        df_ = preprocessing(df, '17', '18', '19', '20', features)
        self.a_dtree.n_features_ = len(features)

        prev = self.a_dtree.predict(df_)

        ans = '未検出'
        if prev == 1: ans = 'い'; print('i')
        elif prev == 0:
            features = ['0_12']
            df_ = preprocessing(df, '8', '12', '16', '20', features)
            self.b_dtree.n_features_ = len(features)

            prev = self.b_dtree.predict(df_)

            if prev == 0: ans = 'あ'
            else: ans = 'う'
        else:
            features = ['theta2', 'a3124', 'a4312', 'a8124', 'd12']
            df_ = preprocessing(df, '3', '4', '8', '12', features)
            self.c_dtree.n_features_ = len(features)

            prev = self.c_dtree.predict(df_)

            if prev == 0: ans = 'え'
            else: ans = 'お'
        
        return ans


    def main(self, image):
        results = self.holistic.process(image)

        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        right_pos = results.right_hand_landmarks
        left_pos = results.left_hand_landmarks
        
        right_ans = left_ans = '未検出'
        if not right_pos is None: right_ans = self.detection(list(right_pos.landmark))                      
        if not left_pos is None: left_ans = self.detection(list(left_pos.landmark))

        h, w, _ = image.shape
        print(h, w)
        image = cv2pil(image)
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), right_ans, (255,255,255), font=ImageFont.truetype('C:/Windows/Fonts/msgothic.ttc', 30))
        draw.text((w-100, 0), left_ans, (255,255,255), font=ImageFont.truetype('C:/Windows/Fonts/msgothic.ttc', 30))
        image = pil2cv(image)

        return image

    def norm_mp_pos(self, pos):
        d = []
        base_x = base_y = 0

        for i in range(21):
            if i == 0: 
                base_y = pos[i].y
                base_x = pos[i].x
            x = pos[i].x-base_x
            y = pos[i].y-base_y
            d.append(x)
            d.append(y)

        s = []
        for i in range(21):
            s.append(str(i)+'x')
            s.append(str(i)+'y')

        df = pd.DataFrame([d], columns=s)
        #row_df = df.copy()
    
        # 角度補正
        y = df['9y'].values
        x = df['9x'].values

        rad = np.arctan2(y, x)

        df['rad'] = rad

        r = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        r = r.reshape((2, 2))

        for j in range(21):
            df[[str(j)+'x', str(j)+'y']] = df[[str(j)+'x', str(j)+'y']] @ r

        return df


class FrontEnd(object):
    def __init__(self):
        self.backend = BackEnd()

    def main(self, cap):
        while True:
            start = time()

            if cv2.waitKey(1) & 0xFF == ord('q'): break

            ret, image = cap.read()
            if not ret: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = self.backend.main(image)
            print(time()-start)

            cv2.imshow('frame', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


        cv2.destroyAllWindows()

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)

    FrontEnd().main(capture)

    capture.release()
    #cv2.destroyAllWindows()



    