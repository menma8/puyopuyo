import cv2
import math 
import numpy as np
import sys
import matplotlib.pyplot as plt

#座標を検出するための関数
def onMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

#直線を検出し、角度を計算するための関数
def read_degree(frame,show_flag=0):
    
    # 指定した四角形の座標
    x1, y1 = 374,556
    x2, y2 = 395,579

    # 四角形の部分を切り出す
    roi = frame[y1:y2, x1:x2]

    # 拡大する（2倍に拡大例）
    scale = 2
    roi_resized = cv2.resize(roi, (roi.shape[1] * scale, roi.shape[0] * scale), interpolation=cv2.INTER_LINEAR)

    #グレースケールに変換
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

    #二値化
    th = 235
    _, img_th = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)

    #エッジ検出
    edges = cv2.Canny(img_th, 50, 10, apertureSize=3)

    # ハフ変換による線分検出
    minLineLength = 5
    maxLineGap = 5
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 10, minLineLength,maxLineGap
    )

    #線の描写および角度の検出
    degs = [] 
    roi_with_lines = roi_resized.copy()
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]

            #線を描写するやつ
            cv2.line(roi_with_lines,(x1,y1),(x2,y2),(255,0,0),2)

            deg = math.degrees(math.atan2(y2-y1, x2-x1))
            degs.append(abs(deg))

    #デバッグ用
    if show_flag:
        cv2.imshow(window_name, roi_resized)
        cv2.waitKey(0)
        cv2.imshow(window_name, edges)
        cv2.waitKey(0)
        cv2.imshow(window_name, roi_with_lines)
        cv2.waitKey(0)


    #平均角度を返す
    if degs:
        return sum(degs)/len(degs)
    return 0

def time_to_frame(minutes: int, seconds: float, fps: int = 30) -> int:
    """
    Calculate the frame number for a given time in a video.

    Args:
        minutes (int): The number of minutes.
        seconds (float): The number of seconds (including decimal seconds).
        fps (int): The frame rate of the video (default is 30).

    Returns:
        int: The corresponding frame number.
    """
    total_seconds = minutes * 60 + seconds
    frame_number = int(total_seconds * fps)
    return frame_number

#色の変化を検出（差は+-10）
def check_color_change(frame, x, y, prev_color, tolerance=30):
    """
    指定座標の色が前回の色と異なるかを判定する。
    RGBそれぞれに対し±toleranceの誤差を許容。

    Args:
        frame: 現在のフレーム（NumPy配列）。
        x, y: チェックするピクセルの座標。
        prev_color: 前回の色 [B, G, R]。
        tolerance: 許容する誤差の範囲。

    Returns:
        bool: 色が変化していればTrue、それ以外はFalse。
    """
    current_color = frame[y, x]
    
    # 各RGB値の差を計算
    diff = np.abs(current_color.astype(int) - prev_color.astype(int))

    if not all(diff <= tolerance):
        print("trigger_flag False")
        print(f"now {current_color}")
        print(f"diff is {diff}")
    
    # すべての差が許容範囲内か確認
    return not all(diff <= tolerance)


#盤面を配列にするやつ
def scan_field(img):

    # マスのサイズと数
    num_rows = 12
    num_cols = 6
    cell_height = img.shape[0] // num_rows
    cell_width = img.shape[1] // num_cols

    # 色の範囲（BGR形式）
    colors = {
        'ozyama':([200,200,200],[230,230,230]),
        'green': ([0, 200, 0], [160, 255, 180]),
        'yellow': ([0, 190, 100], [155, 255, 255]),
        'red': ([0, 0, 170], [220, 220, 255]),
        'blue': ([100, 0, 0], [255, 190, 130]),
        'purple': ([80, 0, 90], [255, 210, 255]),
        'empty': ([0, 0, 0], [50, 50, 50])  # 空白の色を追加
    }

    # ぷよぷよ盤面配列を初期化
    puyo_board = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

    # 平均色のRGB値を格納するための配列を初期化
    average_colors = [[(0, 0, 0) for _ in range(num_cols)] for _ in range(num_rows)]

    # 結果を表示するための空白の画像を作成
    output_img = np.zeros_like(img)

    # 色の対応表（1:赤, 2:青, 3:緑, 4:黄色, 5:紫）
    color_map = {
        1: (0, 0, 255),   # 赤 (BGR)
        2: (255, 0, 0),   # 青
        3: (0, 255, 0),   # 緑
        4: (0, 255, 255), # 黄色
        5: (255, 0, 255), # 紫
        0: (0, 0, 0),      # 空 (黒)
        6: (255,255,255) #おじゃま
    }

    # マスごとに中央付近の2箇所のピクセルの色を取得して平均を計算
    for row in range(num_rows):
        for col in range(num_cols):
            # 各マスの中央の少し上と下のピクセルの座標を計算
            center_x = col * cell_width + cell_width // 2
            center_y1 = row * cell_height + cell_height // 4  # 上の方
            center_y2 = row * cell_height + (cell_height * 3 // 4)  # 下の方

            # 指定した2つのピクセルの色を取得（BGR形式）
            pixel_color1 = img[center_y1, center_x]
            pixel_color2 = img[center_y2, center_x]

            # 2つのピクセルの色の平均を計算
            avg_pixel_color_bgr = (pixel_color1.astype(int) + pixel_color2.astype(int)) // 2

            # BGRをRGBに変換
            avg_pixel_color_rgb = avg_pixel_color_bgr[::-1]

            # 平均色をRGB形式で保存
            average_colors[row][col] = tuple(avg_pixel_color_rgb)

            # 各色について該当するか確認
            for color_name, (lower, upper) in colors.items():
                lower_bound = np.array(lower, dtype="uint8")
                upper_bound = np.array(upper, dtype="uint8")

                # 平均色（BGR）が範囲内にあるか確認
                if np.all(avg_pixel_color_bgr >= lower_bound) and np.all(avg_pixel_color_bgr <= upper_bound):
                    if color_name == 'red':
                        puyo_board[row][col] = 1
                    elif color_name == 'blue':
                        puyo_board[row][col] = 2
                    elif color_name == 'green':
                        puyo_board[row][col] = 3
                    elif color_name == 'yellow':
                        puyo_board[row][col] = 4
                    elif color_name == 'purple':
                        puyo_board[row][col] = 5
                    elif color_name == 'ozyama':
                        puyo_board[row][col] = 6
                    else:
                        puyo_board[row][col] = 0
                    break

            # 数字に対応する色を取得し、そのマスに色を描画
            cell_color = color_map[puyo_board[row][col]]
            cv2.rectangle(output_img, (col * cell_width, row * cell_height),
                          ((col + 1) * cell_width, (row + 1) * cell_height),
                          cell_color, -1)

            # # 元の画像にピクセル取得位置に円を描画（赤色の円）
            # cv2.circle(img, (center_x, center_y1), 5, (0, 0, 255), -1)  # 上の点
            # cv2.circle(img, (center_x, center_y2), 5, (0, 0, 255), -1)  # 下の点

    # # 平均色のRGB値を表示
    # print("\n平均色のRGB値:")
    # for row in average_colors:
    #     print(row)

    # print("\n")


    #✖を黒くする
    #要修正！！！！

    puyo_board[0][2] = 0

    cell_width = img.shape[1] // num_cols
    cell_height = img.shape[0] // num_rows

    # 対応するマスに黒色を描画
    cv2.rectangle(
        output_img,
        (2 * cell_width, 0 * cell_height),  # 左上の座標
        ((2 + 1) * cell_width, (0 + 1) * cell_height),  # 右下の座標
        (0, 0, 0),  # 黒色
        -1  # 塗りつぶし
    )



    # 人間用再現盤面を画像として保存
    output_path = 're_puyo_board_output.png'
    cv2.imwrite(output_path, output_img)
    # print(f"処理結果を画像として保存しました: {output_path}")


    # 結果を表示する（VSCode向け）
    # cv2.imshow("Puyo Board Output", output_img)

    #点を打った画像を保存する
    output_name = 're_puyo_board_output2.png'
    cv2.imwrite(output_name, img)

    return puyo_board


# 連鎖計算とシミュレーション関数
def puyo_simulation(initial_board):
    """
    初期盤面を受け取り、連鎖とスコアを計算するシミュレーションを実行します。

    Args:
        initial_board (list[list[int]]): 初期盤面の2次元リスト

    Returns:
        tuple: (最終連鎖数, 最終スコア, おじゃまぷよの数)
    """
    # 初期盤面をNumPy配列に変換
    board = np.array(initial_board)

    # 色番号と名前の対応辞書
    color_names = {
        1: "赤",
        2: "青",
        3: "緑",
        4: "黄",
        5: "紫",
        6: "おじゃま",
    }

     # ボーナス計算関数
    def calculate_bonus(chain, connected, colors):
        # 連鎖ボーナス

        print(f"chain is {chain}")

        if chain <= 6:
            chain_bonus = [0, 8, 16, 32, 64, 96, 128][chain-1]
        else:
            chain_bonus = 128 + (chain - 7) * 32

        print(f"chain bonus is {chain_bonus}")

        # 連結ボーナス

        print(f"connect is {connected}")

        if connected <= 3:
            connect_bonus = 0
        elif connected >= 11:
            connect_bonus = 10
        else:
            connect_bonus = [0, 0, 0, 0, 2, 3, 4, 5, 6, 7][connected-1]

        print(f"connected bonus is {connect_bonus}")

        # 色数ボーナス

        print(f"color is {colors}")
        color_bonus = [0, 3, 6, 12][colors - 1]

        print(f"color bonus is {color_bonus}")

        return chain_bonus + connect_bonus + color_bonus
    


    # 方向ベクトル (上下左右)
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    # 盤面を表示する関数
    def print_board():
        print("\n".join(["".join([str(cell) if cell != 0 else '.' for cell in row]) for row in board]))
        print("-" * 12)

    # 連結したぷよを探すDFS
    def dfs(x, y, color, visited):
        stack = [(x, y)]
        connected = []
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            connected.append((cx, cy))
            for i in range(4):
                nx, ny = cx + dx[i], cy + dy[i]
                if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1] and (nx, ny) not in visited:
                    if board[nx, ny] == color:
                        stack.append((nx, ny))
        return connected

     # 連鎖と得点計算
    total_score = 0
    total_chain = 0
    while True:
        visited = set()
        to_clear = []
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] != 0 and board[i, j] != 6 and (i, j) not in visited:
                    connected = dfs(i, j, board[i, j], visited)
                    if len(connected) >= 4:
                        to_clear.append((board[i, j], connected))

        if not to_clear:
            break
        
        print("\n消去前の盤面:")
        print_board()  # 消去前の盤面を表示

        # 消去時の色ごとのカウント
        color_count = {}
        for color, group in to_clear:
            if color not in color_count:
                color_count[color] = 0
            color_count[color] += len(group)
            for x, y in group:
                board[x, y] = 0  # 消去

        print("\n消去後の盤面")
        print_board()  # 消去後の盤面を表示

        # 色ごとの消去数を表示
        print(f"\n連鎖 {total_chain + 1} の結果:")
        for color, count in color_count.items():
            print(f"  色: {color_names[color]} - 消去数: {count}")

        total_chain += 1

        # ボーナス計算
        X = sum(color_count.values()) * 10
        A = calculate_bonus(total_chain, max(color_count.values()), len(color_count))
        total_score += X * A

        # 落下処理
        for col in range(board.shape[1]):
            new_col = [board[row, col] for row in range(board.shape[0]) if board[row, col] != 0]
            # 下からぷよを詰める処理
            for row in range(board.shape[0]):
                if row < len(new_col):
                    board[board.shape[0] - 1 - row, col] = new_col[len(new_col) - 1 - row]
                else:
                    board[board.shape[0] - 1 - row, col] = 0

        print("\n落下後の盤面:")
        print_board()  # 落下後の盤面を表示
        print("\n")


    ojama_puyos = total_score // 70

    return total_chain, total_score, ojama_puyos



#同一ディレクトリの中に入れてね
file_path = "参考動画.mp4"
#コマを送るスピードの遅延値
delay = 30
window_name = "open_cv"
cap = cv2.VideoCapture(file_path)

trigger_flag = False
color_check_position = (516, 122)
previous_color = None


#内部処理用解説用
fr = 0

#簡易フレーム(4,×のみ)
four = 30*(60*3+ 26)
batu = 657
kibou = batu

#直線検出を見せるためのフラッグ
show_flag = 0
#座標を取得するためのフラッグ
show_coordinate_flag = 0
#時間計算するための変数
time_calculate_flag = 0
m = 0  # minutes
s = 21.83  # seconds

#内部処理解説用
if any([show_flag, show_coordinate_flag,time_calculate_flag]):
    while True:
        ret, frame = cap.read()
        fr += 1
        if fr == kibou:
            break

    if show_coordinate_flag:
        cv2.imshow('sample', frame)
        cv2.setMouseCallback('sample', onMouse)
        cv2.waitKey(0)

    if show_flag:
        degree = read_degree(frame,show_flag)
        print(f"Detected degree: {degree}")
        cv2.destroyAllWindows()

    if  time_calculate_flag:
    
        # Calculate the frame number
        answer = time_to_frame(m, s)
        print(f"The frame number for {m} minutes and {s} seconds is: {answer}")
        
    sys.exit()

#現在のフレーム数
now_frame = 1

chains = 0
score = 0
ojama_puyos = 0

#メインループ
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rensa_txt = "chains:  "
    if chains:
        rensa_txt += str(int(chains))
    score_txt = "score:  "
    if score:
        score_txt += str(int(score))
    ojama_txt = "Ojama Puyo:  "
    if ojama_puyos:
        ojama_txt += str(int(ojama_puyos))

    cv2.putText(frame,rensa_txt,(498,505),
                cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0),thickness= 2)
    cv2.putText(frame,score_txt,(498,535),
                cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0),thickness= 2)
    cv2.putText(frame,ojama_txt,(498,565),
                cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0),thickness= 2)

    if not trigger_flag:
        degree = read_degree(frame)
        if 42 <= degree <= 46:
            print(f"Detected degree in range: {degree}")
            
             # 矩形領域を切り出し
            x1, y1 = 226, 113
            x2, y2 = 467, 548
            cropped_frame = frame[y1:y2, x1:x2]

            # # 切り出した領域を表示
            # cv2.imshow("Cropped Frame", cropped_frame)

            # 必要なら画像を保存
            cut_field = f"cut_field_{now_frame}.png"
            cv2.imwrite(cut_field, cropped_frame)
            print(f"Saved cropped frame to {cut_field}")

            cropped_frame = frame[y1:y2, x1:x2]

            bord = scan_field(cropped_frame)
            for row in bord:
                print(row)

            # シミュレーション実行
            chains, score, ojama_puyos = puyo_simulation(bord)

            # 結果を出力
            print(f"\n最終結果: {chains} 連鎖")
            print(f"スコア: {score} 点")
            print(f"おじゃまぷよ: {ojama_puyos} 個")

    
            trigger_flag = True
            previous_color = frame[color_check_position[1], color_check_position[0]]
            print("trigger_flag True")
            print(previous_color)



    else:
        if check_color_change(frame, *color_check_position, previous_color, tolerance=10):
            print("Color changed! Resetting flag.")
            
            trigger_flag = False

    cv2.imshow(window_name, frame)
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
