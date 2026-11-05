# Header 是发布时间；通道时间取有效 SENS 帧的接收时刻，DF/TF=0 不更新。
# valid 表示连接可用且已有有效采样；新鲜度须另查通道时间。
std_msgs/Header header
bool temperature_valid
uint64 temperature_seq
time temperature_stamp
# TAVG 的 Kalman 值，°C；无读数时 NaN。
float64 temp_filtered_c
bool distance_valid
uint64 distance_seq
time distance_stamp
# 原始测距 r=D/1000，m，不扣安装偏移；无读数时 NaN。
float64 distance_m
