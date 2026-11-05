import csv
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from sklearn.linear_model import LinearRegression


class MaskProcessor:
    def __init__(self, kernel_size=15, min_ratio=0.1):
        self.kernel_size = kernel_size
        self.min_ratio = min_ratio

    def denoise(self, mask, class_id):
        mask_id = (mask == class_id).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        filtered_mask = cv2.morphologyEx(mask_id, cv2.MORPH_CLOSE, kernel, iterations=2)
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        labeled = label(filtered_mask, connectivity=2)
        regions = regionprops(labeled)
        total_class_area = sum(r.area for r in regions)
        filtered_mask = remove_small_objects(filtered_mask.astype(bool),
                                             total_class_area * self.min_ratio).astype(np.uint8)
        return filtered_mask * class_id

    def get_contact_points(self, mask_a, mask_b):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_a = cv2.dilate(mask_a, kernel)
        dilated_b = cv2.dilate(mask_b, kernel)
        contact_area = cv2.bitwise_and(dilated_a, dilated_b)
        contours, _ = cv2.findContours(contact_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.array([])
        all_points = np.vstack([c.reshape(-1, 2) for c in contours])
        points = np.unique(all_points, axis=0)
        sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
        sorted_points = points[sorted_indices]
        return sorted_points


class SpineFitter:
    def __init__(self, method='ols'):
        self.method = method
        self.slope = None
        self.intercept = None
        self.start_point = None
        self.end_point = None

    def fit(self, points, start=None, end=None):
        x = points[:, 1].reshape(-1, 1)
        y = points[:, 0]
        if start is None:
            start = x.min()
        if end is None:
            end = x.max()
        x_pred = np.linspace(start, end, 100).reshape(-1, 1)
        if self.method == 'ols':
            ols = LinearRegression()
            ols.fit(x, y)
            self.slope = ols.coef_[0]
            self.intercept = ols.intercept_
            y_pred = ols.predict(x_pred)
        self.start_point = np.array((y_pred[0], x_pred[0][0]))
        self.end_point = np.array((y_pred[-1], x_pred[-1][0]))
        return self.start_point, self.end_point, self.slope, self.intercept

    def get_point_from_y(self, y):
        x = self.slope * y + self.intercept
        return np.array((x, y))

    def divide_spine(self, keypoints, scores):
        def line_intersection(point1, point2, point3, point4):
            x1, y1 = point1
            x2, y2 = point2
            x3, y3 = point3
            x4, y4 = point4
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                return None
            num_x = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
            num_y = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
            x = num_x / denom
            y = num_y / denom
            return np.array((x, y))
        c7 = self.start_point
        t3 = line_intersection(self.start_point, self.end_point, keypoints[1], keypoints[2])
        t6 = line_intersection(self.start_point, self.end_point, keypoints[3], keypoints[4])
        t7 = line_intersection(self.start_point, self.end_point, keypoints[5], keypoints[6])
        l2 = line_intersection(self.start_point, self.end_point, keypoints[7], keypoints[8])
        l5 = line_intersection(self.start_point, self.end_point, keypoints[9], keypoints[10])
        if any(point is None for point in (t3, t6, t7, l2, l5)):
            raise ValueError("关键点连线不能确定脊柱分段")
        a = (t3[1] - c7[1])/3
        b = (t7[1] - t3[1] - a)/3
        d = (l5[1] - l2[1])/3
        c = (l2[1] - t7[1] - b - 2*d)/4
        t1 = self.get_point_from_y(c7[1] + a)
        t2 = self.get_point_from_y(c7[1] + 2*a)
        t4 = self.get_point_from_y(t3[1] + a)
        t5 = self.get_point_from_y(t3[1] + a + b)
        t8 = self.get_point_from_y(t7[1] + b)
        t9 = self.get_point_from_y(t7[1] + b + c)
        t10 = self.get_point_from_y(t7[1] + b + 2*c)
        t11 = self.get_point_from_y(t7[1] + b + 3*c)
        t12 = self.get_point_from_y(t7[1] + b + 4*c)
        l1 = self.get_point_from_y(l2[1] - d)
        l3 = self.get_point_from_y(l2[1] + d)
        l4 = self.get_point_from_y(l2[1] + 2*d)
        return { "C7": c7, "T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6,
                "T7": t7, "T8": t8, "T9": t9, "T10": t10, "T11": t11, "T12": t12,
                "L1": l1, "L2": l2, "L3": l3, "L4": l4, "L5": l5,
        }


class DepthMapper:
    def __init__(self):
        pass

    def locate_point(self, depth_map, depth_scale, start_pixel, direction, distance, intrinsics):
        pixel = np.asarray(start_pixel) / depth_scale
        if not np.isfinite(pixel).all() or np.any(pixel < 0):
            return None
        u, v = pixel.astype(int)
        if (u < 0 or u >= depth_map.shape[1] or
                v < 0 or v >= depth_map.shape[0]):
            return None
        depth = depth_map[v, u]
        if depth <= 0 or not np.isfinite(depth):
            return None
        # 深度图按毫米计算，空间距离和同身寸按米计算。
        z = depth * 0.001
        x = (u - intrinsics['cx'] / depth_scale) * z / (intrinsics['fx'] / depth_scale)
        y = (v - intrinsics['cy'] / depth_scale) * z / (intrinsics['fy'] / depth_scale)
        start_point = np.array([x, y, z])
        direction = np.array(direction)
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm == 0:
            return None
        direction = direction / norm
        end_point = start_point + direction * distance
        x_, y_, z_ = end_point
        u_ = int((x_ * (intrinsics['fx'] / depth_scale) / z_) + (intrinsics['cx'] / depth_scale))
        v_ = int((y_ * (intrinsics['fy'] / depth_scale) / z_) + (intrinsics['cy'] / depth_scale))
        end_pixel = np.array([u_, v_])
        height, width = depth_map.shape
        if 0 <= end_pixel[0] < width and 0 <= end_pixel[1] < height:
            return end_pixel * depth_scale
        else:
            return None


class AcupointLocator:
    def __init__(self, depth_mapper):
        self.depth_mapper = depth_mapper

    def locate_acupoints(self, depth_map, depth_scale, acupoints_info, spine_dict, slope, cun_unit, intrinsics):
        acupoints_dict = {}
        for acupoint in acupoints_info:
            if acupoint['vertical_position'] not in spine_dict.keys():
                continue
            if acupoint['is_double']:
                start_pixel = spine_dict[acupoint['vertical_position']]
                direction1 = (-1 / slope, 1, 0)
                direction2 = (1 / slope, -1, 0)
                distance = acupoint['horizontal_position'] * cun_unit
                point1 = self.depth_mapper.locate_point(depth_map, depth_scale, start_pixel, direction1, distance, intrinsics)
                point2 = self.depth_mapper.locate_point(depth_map, depth_scale, start_pixel, direction2, distance, intrinsics)
                if point1 is None or point2 is None:
                    continue
                if slope > 0:
                    acupoints_dict[acupoint['acupoint_name'] + '_L'] = point1
                    acupoints_dict[acupoint['acupoint_name'] + '_R'] = point2
                else:
                    acupoints_dict[acupoint['acupoint_name'] + '_L'] = point2
                    acupoints_dict[acupoint['acupoint_name'] + '_R'] = point1
            else:
                acupoints_dict[acupoint['acupoint_name']] = spine_dict[acupoint['vertical_position']]
        return acupoints_dict


class AcupointDetectionPipeline:
    def __init__(self):
        self.mask_processor = MaskProcessor()
        self.spine_fitter = SpineFitter()
        self.depth_mapper = DepthMapper()
        self.acupoint_locator = AcupointLocator(self.depth_mapper)

    def process_image(self, rgb, mask, depth_map, keypoints, scores, depth_scale, cun_unit, intrinsics, plot=True):
        denoised_mask_4 = self.mask_processor.denoise(mask, 4)
        denoised_mask_5 = self.mask_processor.denoise(mask, 5)
        points = self.mask_processor.get_contact_points(denoised_mask_4, denoised_mask_5)
        if points.size == 0:
            return rgb.copy(), {}
        point_start, point_end, slope, intercept = self.spine_fitter.fit(
            points, start=keypoints[0][1]
        )
        if not np.isfinite(slope) or slope == 0:
            raise ValueError("脊柱斜率不能用于当前同身寸方向计算")
        spine_dict = self.spine_fitter.divide_spine(keypoints, scores)
        acupoints_dict = self.acupoint_locator.locate_acupoints(
            depth_map, depth_scale, self.acupoints_info, spine_dict, slope, cun_unit, intrinsics
        )
        if plot:
            img_plot = self._visualize_results(
                rgb, points, point_start, point_end, keypoints, spine_dict, acupoints_dict
            )
        else:
            img_plot = rgb.copy()
        return img_plot, acupoints_dict

    def _visualize_results(self, rgb, points, point_start, point_end, keypoints, spine_dict, acupoints_dict):
        img_plot = rgb.copy()
        cv2.line(img_plot, point_start.astype(np.uint16), point_end.astype(np.uint16), (0, 255, 0), 2)
        keypoint_names = ["C7", "left_acromion", "right_acromion", "left_armpit", "right_armpit",
                          "left_ais", "right_ais", "left_waist", "right_waist", "left_hip", "right_hip"]
        for i in range(1, 11):
            cv2.circle(img_plot, keypoints[i].astype(np.uint16), 3, (255, 255, 0), -1)
            cv2.putText(img_plot, keypoint_names[i], keypoints[i].astype(np.uint16),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        for _key, _point in spine_dict.items():
            cv2.circle(img_plot, _point.astype(np.uint16), 3, (0, 255, 255), -1)
            cv2.putText(img_plot, _key, _point.astype(np.uint16), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        for _key, _point in acupoints_dict.items():
            cv2.circle(img_plot, _point.astype(np.uint16), 3, (0, 0, 255), -1)
            cv2.putText(img_plot, _key, _point.astype(np.uint16) - 5,
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        return img_plot

    def load_acupoints_info(self, acupoints_file):
        self.acupoints_info = []
        with open(acupoints_file, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["horizontal_position"] = float(row["horizontal_position"])
                row["is_double"] = (int(row["is_double"]) == 1)
                self.acupoints_info.append(row)
