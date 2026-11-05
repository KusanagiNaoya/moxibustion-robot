#!/usr/bin/env python3
"""启动艾灸机器人 PyQt 操作台。"""
import sys

import rospy
from PyQt5.QtWidgets import QApplication
from system_integration.panel_acupoint_moxibustion import AcupointMoxibustionPanel


def main():
    rospy.init_node('panel_acupoint_moxibustion', disable_signals=True)
    app = QApplication(sys.argv)
    panel = AcupointMoxibustionPanel()
    panel.show()
    result = app.exec_()
    rospy.signal_shutdown('操作台已关闭')
    return result


if __name__ == '__main__':
    sys.exit(main())
