import numpy as np
import scipy.io as sio

# 데이터 경로
data_path = "meas_gre_dir1.mat"

# .mat 파일 불러오기
mat_data = sio.loadmat(data_path)

# 저장된 변수 확인
print(mat_data.keys())