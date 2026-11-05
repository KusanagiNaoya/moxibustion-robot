# header.stamp 为 Kortex 反馈接收时间，缓存重发不更新；不是硬件采样时间。
# header.frame_id 由 base_frame 标注，不执行坐标转换。
Header header
kinova_controller/KinovaPose pose
kortex_driver/BaseCyclic_Feedback robot_status
