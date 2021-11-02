import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import cv2
from gym.envs.classic_control import CartPoleEnv


class HeadlessCartPoleEnv(CartPoleEnv):

    def _rotate(self, point, rad):
        x, y = point
        x_ = math.cos(rad) * x - math.sin(rad) * y
        y_ = math.sin(rad) * x + math.cos(rad) * y
        return [x_, y_]

    def render(self, mode='rgb_array', width=600, height=400):
        assert mode == 'rgb_array'
        screen_width = 600
        screen_height = 400
        canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8) + 255

        cart_color = (0, 0, 0)
        pole_color = (int(255 * 0.8), int(255 * 0.6), int(255 * 0.4))
        axle_color = (int(255 * 0.5), int(255 * 0.5), int(255 * 0.8))

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART```
        pole_rad = -x[2]

        l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
        axleoffset = cartheight/4.0
        t, b = t + carty, b + carty
        l, r = l + cartx, r + cartx
        canvas = cv2.fillPoly(canvas, np.array([[[l, b], [l, t], [r, t], [r, b]]], dtype=np.int32), color=cart_color)

        l, r, t, b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2

        lb = self._rotate([l, b], pole_rad)
        lt = self._rotate([l, t], pole_rad)
        rb = self._rotate([r, b], pole_rad)
        rt = self._rotate([r, t], pole_rad)

        canvas = cv2.fillPoly(canvas, np.array([[[lb[0] + cartx, lb[1] + carty + axleoffset],
                                       [lt[0] + cartx, lt[1] + carty + axleoffset],
                                       [rt[0] + cartx, rt[1] + carty + axleoffset],
                                       [rb[0] + cartx, rb[1] + carty + axleoffset]]], dtype=np.int32), color=pole_color)
        canvas = cv2.circle(canvas, (int(cartx), int(carty + axleoffset)), int(polewidth/2), color=axle_color, thickness=-1)
        canvas = cv2.line(canvas, (0, carty), (screen_width, carty), cart_color)
        canvas = cv2.flip(canvas, -1)
        return cv2.resize(canvas, (width, height))


if __name__ == '__main__':
    env = HeadlessCartPoleEnv()
    env.reset()
    for i in range(200):
        _, _, done, _ = env.step(env.action_space.sample())
        img = env.render()
        img[..., [0, 2]] = img[..., [2, 0]]
        cv2.imshow('CartPole', img)
        cv2.waitKey(20)
        if done:
            env.reset()
    exit()