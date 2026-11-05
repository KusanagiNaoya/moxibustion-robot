# 位置为米，姿态为单位四元数；时间戳是反馈时间，缓存重发不得更新。
geometry_msgs/PoseStamped pose
# 后端报告和控制的参考点标识，不要求存在同名 TF。
string controlled_frame
# 无反馈、反馈过期、坐标或位姿无效时为 false。
bool valid
