# 坐标系由 AcupointArray.header.frame_id 指定。
string name
bool is_3d
# 图像像素坐标。
float32 u
float32 v
# 三维坐标，单位 m；is_3d 为 false 时不作为空间目标使用。
geometry_msgs/Point point
# 负值表示无有效深度。
float32 confidence
