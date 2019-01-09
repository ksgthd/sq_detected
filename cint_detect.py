#!/usr/local/bin/python
#! -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
from datetime import datetime
import sys
from tqdm import tqdm
now_time_str = datetime.now().strftime("%Y%m%d_%H-%M-%S")
def contrast(image, a, mid):
  lut = [ np.uint8(255.0 / (1 + math.exp(-a * (i - mid) / 255.))) for i in range(256)]
  result_image = np.array( [ lut[value] for value in image.flat], dtype=np.uint8 )
  result_image = result_image.reshape(image.shape)
  return result_image

# 指定した画像(path)の物体を検出し、外接矩形の画像を出力
def detect_contour(path, name):
  MidLine = []
  All = []
  LIST = [[20, 100], [50, 100]]
  for P in range(100, 200, 12):
      LIST.append([P, 150])

  for P in LIST:
      mid = P[0]
      cntrst_level = P[1]
      # 画像を読込
      src = cv2.imread(path, cv2.IMREAD_COLOR)
      # グレースケール画像へ変換
      gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
      gray = contrast(gray, cntrst_level, mid)
      retval, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO | cv2.THRESH_OTSU)
      cv2.imwrite('contrast/' + name.split('.')[0] + '_contrast_' + str(cntrst_level) +'.png', gray)

      # 輪郭を抽出
      image, contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

      # 矩形検出された数（デフォルトで0を指定）
      detect_count = 0
      rects = []
      for i in range(0, len(contours)):
        # 輪郭の領域を計算
        area = cv2.contourArea(contours[i])
        # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
        if area < 1e2 or 1e5 < area:
          continue
        rect = contours[i]
        x, y, w, h = cv2.boundingRect(rect)
        if w > 40:
            rects.append([x, y, w, h])
            All.append([x, y, w, h])
      x_sort = []
      for i, r in enumerate(rects):
        if not x_sort:
          x_sort.append(r)
        else:
            check = 0
            for j, sort_r in enumerate(x_sort):
                if r[0] < sort_r[0]:
                    x_sort.insert(j, r)
                    check = 1
                    break
            if not check:
                x_sort.append(r)
      neighbor = []
      for now, r in enumerate(x_sort):
          for i in range(1, 5):
              if now + i < len(x_sort):
                  sa = (x_sort[now + i][0] - (r[0] + r[2]))
                  if sa < 30 and sa > -10:
                      neighbor.append([r, x_sort[now + i]])
                      x, y, w, h = r
                      cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
                      x, y, w, h = x_sort[now+i]
                      cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
      neighbor_line = []
      for nb in neighbor:
          for nb_ in nb:
              exst_check = 0
              for i, nb_already in enumerate(neighbor_line):
                  if nb_ == nb_already:
                      exst_check = 1
                      break
              if not exst_check:
                  insert_check = 0
                  for i, nb_already in enumerate(neighbor_line):
                      if nb_already[0] < nb_[0]:
                          neighbor_line.insert(i, nb_)
                          insert_check = 1
                          break
                  if not insert_check:
                      neighbor_line.append(nb_)
      cv2.imwrite('detected_Image/' + name.split('.')[0] + '_detected_Xneighbor_' + str(mid) +'_' + str(cntrst_level) + '.png', src)
      for nb in neighbor:
          sa = nb[1][0] - (nb[0][0] + nb[0][2])
          x_l = nb[0][0] + nb[0][2]
          y_l = nb[0][1] + nb[0][3]
          x_r = nb[1][0]
          y_r = nb[1][1]
          MidLine.append([nb[1][0] - int(sa/2), [x_l, y_l, x_r, y_r]])
  if len(MidLine) != 0:
      MidLine_X_max = 0
      MidLine_X_min = 999999
      NearestMid = {}
      for x_M in MidLine:
          if MidLine_X_max < x_M[0]:
              MidLine_X_max = x_M[0]
          if MidLine_X_min > x_M[0]:
              MidLine_X_min = x_M[0]
          if NearestMid == {}:
              NearestMid[x_M[0]] = [x_M[0]]
          else:
              check = 0
              for key in NearestMid:
                  if abs(sum(NearestMid[key])/len(NearestMid[key]) - x_M[0]) < 15:
                      NearestMid[key].append(x_M[0])
                      check = 1
                      break
              if not check:
                  NearestMid[x_M[0]] = [x_M[0]]
      #一番右の仕切り線と一番左の仕切り線の間の距離(x)の導出
      if len(MidLine) > 1:
          MidLineWidth = MidLine_X_max - MidLine_X_min
      else:
          MidLineWidth = int(src.shape[1]*0.8)

      #横幅での判定
      src = cv2.imread(path, cv2.IMREAD_COLOR)
      # グレースケール画像へ変換
      gray_ok = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
      gray_no_w = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
      gray_no = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
      OK_squares = []
      No_squares = []
      OK_x_area = []#仕切り線が通っていて欲しくないエリア(x)のリスト
      for r in All:
          width = r[2]*10
          if abs(MidLineWidth - width) < MidLineWidth*0.25:
              OK_squares.append(r)
              x, y, w, h = r
              cv2.rectangle(gray_ok, (x, y), (x + w, y + h), (0, 255, 0), 2)
              #仕切り線が通っていてほしくないエリアの更新
              ThisArea = [x, w]
              if OK_x_area == []:
                  OK_x_area.append([ThisArea])
              else:
                  check_all = 0
                  for index, x_area in enumerate(OK_x_area):
                    inside = 0
                    check = 0
                    for i, x_area2 in enumerate(x_area):
                        if x_area2[0] >= x and sum(x_area2) <= x + w:
                            OK_x_area[index].insert(i, ThisArea)
                            check = 1
                            break
                        elif x_area2[0] < x and sum(x_area2) > x + w:
                            inside = 1
                    if check:
                        check_all = 1
                        break
                    if inside and not check:
                        OK_x_area[index].append(ThisArea)
                        check_all = 1
                        break
                  if not check_all:
                    OK_x_area.append([ThisArea])
          else:
              No_squares.append(r)
              x, y, w, h = r
              cv2.rectangle(gray_no, (x, y), (x + w, y + h), (0, 255, 0), 2)
      OK_MidLines = []
      for key in NearestMid:
          MidLine_x = int(sum(NearestMid[key])/len(NearestMid[key]))
          check = 0
          for x_area in OK_x_area:
              if len(x_area) == 1:
                  area_width = 0.9 * x_area[-1][1]
                  area_x = x_area[-1][0] + 0.05*x_area[-1][1]
              else:
                  area_width = x_area[-1][1]
                  area_x = x_area[-1][0]
              if area_x < MidLine_x and (area_x + area_width) > MidLine_x:
                  check = 1
                  break
          if check:
              cv2.line(gray_no, (MidLine_x, 300), (MidLine_x, 1400), 0, thickness=1, lineType=cv2.LINE_4)
              continue
          OK_MidLines.append(MidLine_x)
      cv2.imwrite(
          'detected_Image/' + name.split('.')[0] + '_detected_1005_NO.png',
          gray_no)
      #エリアから外れている合格仕切り線の中から、幅を基準にさらに選定
      OK_MidLines_w = []
      OK_MidLines.sort()
      MidLines_Check_BUFF = [[abs(OK_MidLines[1]-OK_MidLines[0])]]
      Lines_BUFF = [[OK_MidLines[0], OK_MidLines[1]]]
      for i, l in enumerate(OK_MidLines[1:]):
          if i == len(OK_MidLines) -1 :
              break
          for j in range(1, 5):
              if i + j == len(OK_MidLines):
                  break
              width = abs(OK_MidLines[i + j] - l)
              check = 0
              for b, buff in enumerate(MidLines_Check_BUFF):
                  ave_w = sum(buff)/len(buff)
                  if abs(ave_w - width) < ave_w*0.2:
                      MidLines_Check_BUFF[b].append(width)
                      Lines_BUFF[b].append(l)
                      Lines_BUFF[b].append(OK_MidLines[i + j])
                      check = 1
                      break
              if not check:
                  MidLines_Check_BUFF.append([width])
                  Lines_BUFF.append([l, OK_MidLines[i + j]])
      most_confidencial = 999
      width_most = 0
      for w, buff in enumerate(MidLines_Check_BUFF):
          if len(buff) > width_most:
              width_most = len(buff)
              most_confidencial = w
      regular_width = sum(MidLines_Check_BUFF[most_confidencial])/len(MidLines_Check_BUFF[most_confidencial])
      for l in Lines_BUFF[most_confidencial]:
          OK_MidLines_w.append(l)
          cv2.line(gray_ok, (l, 300), (l, 1400), 0, thickness=1, lineType=cv2.LINE_4)
      cv2.imwrite(
          'detected_Image/' + name.split('.')[0] + '_detected_1000_OK.png',
          gray_ok)
      #選定で落ちたもの
      for b, buff in enumerate(Lines_BUFF):
          if b != most_confidencial:
              for l in buff:
                  cv2.line(gray_no_w, (l, 300), (l, 1400), 0, thickness=1, lineType=cv2.LINE_4)
      cv2.imwrite('detected_Image/' + name.split('.')[0] + '_detected_1001_NOw.png', gray_no_w)

      #仕切り線との関係からトリミング枠を決定する前準備
      LinesAreas = []
      OK_MidLines_w = list(set(OK_MidLines_w))
      OK_MidLines_w.sort()
      #仕切り線の補完
      new_Lines = []
      for index, line in enumerate(OK_MidLines_w):
          if index == len(OK_MidLines_w) -1:
              break
          width_c = OK_MidLines_w[index + 1] - line
          if width_c > 1.4*regular_width:
              check = 0
              for i in range(2, 5):
                  if width_c > (i - 0.5)*regular_width and width_c <= (i + 0.5)*regular_width:
                      width_p = int(width_c/i)
                      for j in range(i):
                          new_Lines.append(line + j*width_p)
                      check = 1
                      break
              if not check:
                  new_Lines.append(line)
          else:
              new_Lines.append(line)
      new_Lines.append(OK_MidLines_w[-1])
      OK_MidLines_w = list(set(new_Lines))
      OK_MidLines_w.sort()
      print('OK_MidLines_w', len(OK_MidLines_w))
      for index, l_x in enumerate(OK_MidLines_w):
          if index == 0:
              LinesAreas.append([l_x - regular_width, l_x])
          if index == len(OK_MidLines_w)-1:
              LinesAreas.append([l_x, l_x + regular_width])
              break
          if index == len(OK_MidLines_w)-1:
              break
          LinesAreas.append([l_x, OK_MidLines_w[index + 1]])
      SquaresList = [ [] for no in LinesAreas]
      SquaresList_confirmed = [ [] for no in LinesAreas]
      print('length LinesAreas ', len(LinesAreas), '\tSquaresList', len(SquaresList))
      for index, lines in enumerate(LinesAreas):
          if index == 0:
              for r in OK_squares:
                  if (r[0] + r[2]*0.95) < lines[1] and (r[0] + r[2]) > (lines[1] - 0.08*regular_width) and abs(r[2] - regular_width) < 0.15*regular_width:
                      SquaresList[index].append(r)
          elif index == len(LinesAreas) - 1:
              for r in OK_squares:
                  if (r[0] + r[2]*0.05) > lines[0] and r[0] < (lines[0] + 0.08*regular_width) and abs(r[2] - regular_width) < 0.15*regular_width:
                      SquaresList[index].append(r)
          else:
              for r in OK_squares:
                  if (r[0] + 0.05*r[2]) > lines[0] and (r[0] + r[2]*0.95) < lines[1]:
                      SquaresList[index].append(r)
      for index, rs in enumerate(SquaresList):
          start = LinesAreas[index][0]
          end = LinesAreas[index][1]
          if rs == []:
              cross_squares = []
              for r in No_squares:
                  append_flag = 0
                  if r[0] > start and (r[0] + r[2]) < end:
                      append_flag = 1
                  elif r[0] > start and r[0]  < end and (r[0] + r[2]) > end and r[2] < 3.5*regular_width:
                      append_flag = 1
                  elif r[0] < start and (r[0] + r[2]) < end and (r[0] + r[2]) > start and r[2] < 3.5*regular_width:
                      append_flag = 1
                  if append_flag:
                      cross_squares.append([r[1], r[1] + r[3]])
              if len(cross_squares):
                  y_bottom = sum([y[0] for y in cross_squares])/len(cross_squares)
                  y_height = sum([y[1] for y in cross_squares])/len(cross_squares) - y_bottom
                  SquaresList_confirmed[index] = [int(p) for p in [start, y_bottom, end-start, y_height]]
              else:
                  SquaresList_confirmed[index] = 'nothing'
          else:
              y = sum([r[1] for r in rs])/len(rs)
              height = sum([r[3] for r in rs])/len(rs)
              if index == 0:
                  w0 = sum([r[2] for r in rs])/len(rs)
                  start = int(end - w0)
                  LinesAreas[index][0] = start
                  SquaresList_confirmed[index] = [int(p) for p in [start, y, w0, height]]
              elif index == len(LinesAreas) - 1:
                  w0 = sum([r[2] for r in rs]) / len(rs)
                  end = int(start + w0)
                  LinesAreas[index][1] = end
                  SquaresList_confirmed[index] = [int(p) for p in [start, y, w0, height]]
              else:
                  SquaresList_confirmed[index] = [int(p) for p in [start, y, end - start, height]]
      #確定矩形リストに何も入っていない場合（候補が一つもなかった場合）
      for check in range(3):
          no_nothing = 99
          for index, r in enumerate(SquaresList_confirmed):
              if r == 'nothing':
                  no_nothing = 0
                  start = LinesAreas[index][0]
                  end = LinesAreas[index][1]
                  if index in [i for i in range(2)]:
                      next_r = SquaresList_confirmed[index + 1]
                      next2_r = SquaresList_confirmed[index + 2]
                      if next_r == 'nothing' or next2_r == 'nothing':
                          continue
                      y_h = next2_r[1] - next_r[1]
                      y = next_r[1] - y_h
                      x = next_r[0] - regular_width
                      SquaresList_confirmed[index] = [int(p) for p in [start, y, end-start, next2_r[3]]]
                  elif index in [len(SquaresList_confirmed)-i for i in range(1, 3)]:
                      next_r = SquaresList_confirmed[index - 1]
                      next2_r = SquaresList_confirmed[index - 2]
                      if next_r == 'nothing' or next2_r == 'nothing':
                          continue
                      y_h = next_r[1] - next2_r[1]
                      y = next_r[1] + y_h
                      x = next_r[0] + next_r[2]
                      SquaresList_confirmed[index] = [int(p) for p in [start, y, end-start, next2_r[3]]]
                  else:
                      next_r = SquaresList_confirmed[index + 1]
                      next2_r = SquaresList_confirmed[index + 2]
                      if next_r == 'nothing' or next2_r == 'nothing':
                          continue
                      y_h = next2_r[1] - next_r[1]
                      y = next_r[1] - y_h
                      pre_r = SquaresList_confirmed[index - 1]
                      pre2_r = SquaresList_confirmed[index - 2]
                      if pre_r == 'nothing' or pre2_r == 'nothing':
                          continue
                      y_h = pre_r[1] - pre2_r[1]
                      y2 = pre_r[1] + y_h
                      height = (next2_r[3] + pre2_r[3])/2
                      SquaresList_confirmed[index] = [int(p) for p in [start, (y+y2)/2, end - start, height]]
          if no_nothing:
              break
      height_list = []
      y_list = []
      for rr in SquaresList_confirmed:
          if rr != 'nothing':
              height_list.append(rr[3])
              y_list.append(rr[1])
      FinSquares = [[LinesAreas[index][0], LinesAreas[index][1], int(sum(height_list)/len(height_list)), int(sum(y_list)/len(y_list))] if r == 'nothing' else r for index, r in enumerate(SquaresList_confirmed)]
      src = cv2.imread(path, cv2.IMREAD_COLOR)
      print('length FinSq', len(FinSquares),'length LinesAreas ', len(LinesAreas), '\tSquaresList', len(SquaresList))
      y_top_min = min([r[1] for r in FinSquares])
      for index, r in enumerate(FinSquares):
          x, y, w, h = r
          l1 =int(LinesAreas[index][0])
          l2 =int(LinesAreas[index][1])
          #print(LinesAreas[index])
          cv2.line(src, (l1, 300), (l1, 1400), (0, 0, 0), thickness=1, lineType=cv2.LINE_4)
          cv2.line(src, (l2, 300), (l2, 1400), (0, 0, 0), thickness=1, lineType=cv2.LINE_4)
          cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)
          cv2.putText(src, '(' + str(index + 1) + ')', (x, y_top_min - 50 ), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2, cv2.LINE_AA)
      cv2.imwrite(
          'detected_Image/' + name.split('.')[0] + '_detected_999_Fin.png',
          src)
      src = cv2.imread(path, cv2.IMREAD_COLOR)
      # グレースケール画像へ変換
      gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
      for key in NearestMid:
          MidLine_x = int(sum(NearestMid[key])/len(NearestMid[key]))
          cv2.line(gray, (MidLine_x, 300), (MidLine_x, 1400), 0, thickness=1, lineType=cv2.LINE_4)
      cv2.imwrite('detected_Image/' + name.split('.')[0] + '_detected_1010_MidLine.png', gray)
      #print(sum([r[0] for r in MidLine])/len(MidLine))
      with open('./detect_details_log/10_' + path.split('/')[-1].split('.')[0] + '_' + now_time_str + '.txt', 'w', encoding='utf-8') as f:
          f.write(path.split('/')[-1] + '\n')
          f.write('contrast mid and contrast level LIST(mid-level)\n')
          f.write('\t'.join(['-'.join([ str(ll) for ll in l]) for l in LIST]) + '\n')
          f.write('near(x)\tnear neighbors\n')
          for key in NearestMid:
              f.write(str(key) + '\t' + '\t'.join([ str(l) for l in NearestMid[key]]) + '\n')
  # 各輪郭に対する処理
  for i in range(0, len(contours)):
    # 輪郭の領域を計算
    area = cv2.contourArea(contours[i])
    # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
    if area < 1e2 or 1e5 < area:
      continue

    # 外接矩形
    if len(contours[i]) > 0:
      rect = contours[i]
      x, y, w, h = cv2.boundingRect(rect)
      cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
      # 外接矩形毎に画像を保存
      cv2.imwrite('./piece/' + str(detect_count) + '.jpg', src[y:y + h, x:x + w])
      detect_count = detect_count + 1
  cv2.imwrite('detected_Image/' + name.split('.')[0] + '_detected.png', src)

if __name__ == '__main__':
    import os
    if len(sys.argv) == 1:
        image_dir = './image_sq/'
    else:
        image_dir = './' + sys.argv[1] + '/'
    print(image_dir)
    lists = os.listdir(image_dir)
    for image in tqdm(lists):
        file = image_dir + image
        detect_contour(file, image)
