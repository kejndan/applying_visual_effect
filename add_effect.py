import numpy as np
import cv2
import os
import pickle


class ApplyEffect:
    def __init__(self, path_to_imgs, path_to_query_img,
                 path_to_annotation,  path_to_effect, use_query_img=True,thr_mask=3,
                 width_increase_mask=10, min_match_count=4, percent_match_for_mask=0.6):
        self.path_to_imgs = path_to_imgs
        self.path_to_query_img = path_to_query_img
        self.path_to_annotation = path_to_annotation
        self.thr_mask = thr_mask
        self.width_increase_mask = width_increase_mask
        self.coords_effect = self.read_coord_effect()
        self.effect = self.prepare_effect(path_to_effect)
        self.min_match_count = min_match_count
        self.percent_match_for_mask = percent_match_for_mask
        self.use_query_img = use_query_img
        if self.use_query_img:
            self.shift_coords_effect = self.calc_shift_relative_query()

    def create_mask(self):
        imgs_path = os.listdir(self.path_to_imgs)[:100]
        imgs = []
        for i in imgs_path:
            if '.DS_Store' != i:
                imgs.append(cv2.imread(os.path.join(self.path_to_imgs, i), 0))

        count_not_diff = np.zeros_like(imgs[0])

        for i in range(len(imgs) - 1):
            count_not_diff += imgs[i] == imgs[i + 1]
        count_not_diff = np.where(count_not_diff > int(len(imgs)*self.percent_match_for_mask), count_not_diff, 0)
        mask = self.increase_mask(count_not_diff, self.width_increase_mask)
        cv2.imwrite(f'masks/img/{self.path_to_imgs.split("/")[-1]}.png', mask)
        with open(f'masks/pickle/{self.path_to_imgs.split("/")[-1]}.pickle', 'wb') as f:
            pickle.dump(np.where(mask > 0, 1, 0), f)

    def increase_mask(self, mask, width):
        new_mask = np.zeros(mask.shape)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            new_mask = cv2.drawContours(new_mask, [contour], 0, (255, 255, 255), width)
        return (new_mask + np.where(mask > 0, 255, 0)).astype(np.uint8)

    def read_coord_effect(self):
        coords = []
        with open(self.path_to_annotation, 'r') as f:
            for coord_line in f.readlines():
                x, y = map(int, coord_line.split())
                coords.append([x, y])
        coords = np.array(coords).T.reshape(2,1,2).astype(np.float64)
        return coords

    def apply_mask(self, img):
        with open(f'masks/pickle/{self.path_to_imgs.split("/")[-1]}.pickle', 'rb') as f:
            mask = pickle.load(f)
        return img * mask

    def keypoint_matching(self, des1, des2):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        return good

    def draw_effect(self, img, coords_for_effect):
        for proj_coord in coords_for_effect:
            center = np.int32(proj_coord)
            x1, y1 = center[0] - self.effect.shape[0] // 2, center[1] - self.effect.shape[1] // 2
            x2, y2 = center[0] + self.effect.shape[0] // 2, center[1] + self.effect.shape[1] // 2
            if x1 >= 0 and y1 >= 0 and x2 < img.shape[0] and y2 < img.shape[1]:
                alpha_s = self.effect[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                img[y1:y1 + self.effect.shape[0], x1:x1 + self.effect.shape[1]] \
                    = (alpha_s[..., None] * self.effect[..., :3]
                       + alpha_l[..., None] * img[y1:y1 + self.effect.shape[0], x1:x1 + self.effect.shape[1], :3])

    def calc_shift_relative_query(self):
        path_to_object = os.path.join(self.path_to_query_img, 'object.png')
        path_to_full_img = os.path.join(self.path_to_query_img, 'full_image.png')
        sift = cv2.SIFT_create()

        query = cv2.imread(path_to_object, 0)
        kp_obj, des_obj = sift.detectAndCompute(query, None)
        img = cv2.imread(path_to_full_img, 0)

        kp_full, des_full = sift.detectAndCompute(img, None)
        good_matches = self.keypoint_matching(des_obj, des_full)
        if len(good_matches) > self.min_match_count:
            src_pts = np.float32([kp_obj[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_full[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            pts = np.float32([[0, 0]]).reshape(-1, 1, 2)
            proj_coords = np.int32(cv2.perspectiveTransform(pts, M)[0][0])
            return self.coords_effect[:, 0].T - proj_coords

    def apply_effect(self):
        sift = cv2.SIFT_create()
        if self.use_query_img:
            query = cv2.imread(os.path.join(self.path_to_query_img, 'object.png'), 0)
            kp, des = sift.detectAndCompute(query, None)
        else:
            query = cv2.imread(os.path.join(self.path_to_query_img, 'full_image.png'), 0)
            mask = self.apply_mask(query).astype(np.uint8)
            cv2.imwrite('mask.png', mask)
            kp, des = sift.detectAndCompute(mask, None)

        for img in os.listdir(self.path_to_imgs):

            if '.DS_Store' != img:
                path_to_img = os.path.join(self.path_to_imgs, img)

                if os.path.isfile(path_to_img):
                    img_w_effect = self.__apply_effect_on_image(path_to_img, sift, kp, des)
                    if not os.path.exists(f'output/{self.path_to_imgs.split("/")[-1]}'):
                        os.mkdir(f'output/{self.path_to_imgs.split("/")[-1]}')
                    cv2.imwrite(f'output/{self.path_to_imgs.split("/")[-1]}/{img}', img_w_effect)


    def __apply_effect_on_image(self, path_to_img, sift, kp_query, des_query):
        img = cv2.imread(path_to_img, 0)
        img_rgb = cv2.imread(path_to_img)
        kp, des = sift.detectAndCompute(img, None)
        good_matches = self.keypoint_matching(des_query, des)

        if len(good_matches) > self.min_match_count:
            src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                if self.use_query_img:
                    pts = np.float32([[0,0]]).reshape(-1, 1, 2)
                    dst = np.int32(cv2.perspectiveTransform(pts, M))[:,0]
                    proj_coords = self.shift_coords_effect + dst #
                else:
                    proj_coords = cv2.perspectiveTransform(self.coords_effect, M)[:, 0, :].T
                self.draw_effect(img_rgb, proj_coords)
        return img_rgb

    def prepare_effect(self, path_to_effect):
        ef = cv2.imread(path_to_effect, -1)
        resize_ef = cv2.resize(ef, (200, 200))
        return resize_ef


if __name__ == '__main__':
    ae = ApplyEffect('/Users/adels/PycharmProjects/datasets/nft/shibu',
                     '/Users/adels/PycharmProjects/datasets/nft/shibu/query',
                     'annotation/bored_ape.txt', use_query_img=True,
                     path_to_effect='flare.png')
    # ae.create_mask()
    ae.apply_effect()