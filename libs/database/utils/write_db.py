# Not part of the main flow

import sys, cv2
sys.path.append('/bask/projects/j/jiaoj-3d-vision/Hao/360x/360x_Video_Experiments')
from release.libs.database.utils.db_utils import database
from glob import glob
import os

def get_video_info(video_path):
    try:
        vid = cv2.VideoCapture(video_path)
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        video_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        # video_len = int(video_frames / fps)
        return fps, video_frames #, video_len

    except:
        print("CANNOT OPEN:", video_path)
        return 0, 0

db = database()

db.execute('''CREATE TABLE IF NOT EXISTS videoinfo
               (id INTEGER PRIMARY KEYT,
                # 360_fps INTEGER, 360_frames INTEGER, 
                stereo_fps INTEGER, stereo_frames INTEGER,
                );''')
#                 clip1_cuts INTEGER, clip2_cuts INTEGER,
#                 clip3_cuts INTEGER, clip4_cuts INTEGER,
#                 clip5_cuts INTEGER, clip6_cuts INTEGER,
#  AUTOINCREMEN


idlist = db.get_idlist()

for id in idlist:
    data = db.get_data_from_id(id, get_nested_data=False)
    panoramic_fps, panoramic_frams = get_video_info(os.path.join(data['x360_original'], "360_panoramic.mp4"))
    binocular_fps, binocular_frams = get_video_info(os.path.join(data['stereo_original'], "clip1", "binocular.mp4"))

    string = f"INSERT INTO videoinfo (id, 360_fps, 360_frames, stereo_fps, stereo_frames) " \
             f"VALUES ({data['id']}, {panoramic_fps}, {panoramic_frams}, {binocular_fps}, {binocular_frams})"

    db.execute(string)

db.close()


#
# 在连接到数据库时，可以使用参数来控制连接行为。例如可以指定检测数据库中的数据类型：
# conn = sqlite3.connect('example.db', detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
# 也可以在连接时打开一个内存数据库：
# conn = sqlite3.connect(':memory:')
# # 删除一条数据
# conn.execute("DELETE FROM person WHERE name='Alice'")
# # 更新数据conn.execute("UPDATE person SET age=21 WHERE name='Bob'")