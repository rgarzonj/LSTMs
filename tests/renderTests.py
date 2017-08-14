#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 18:18:21 2017

@author: rubengarzon
"""

from gym.envs.classic_control import rendering
import gym
import pyglet
from rendering import Geom

dddd = None

screen_width = 600
screen_height = 400



carty = 100 # TOP OF CART
polewidth = 10.0
cartwidth = 50.0
cartheight = 30.0


if dddd is None:
    dddd = rendering.Viewer(screen_width, screen_height)
    l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
    axleoffset =cartheight/4.0
    block = rendering.make_polygon([(l,b), (l,t), (r,t), (r,b)],False)

    dddd.add_geom(block)

mode = 'human'


class MyLabel(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=600//2, y=400//2,
                          anchor_x='center', anchor_y='center')
        label.draw()

#window = pyglet.window.Window()
label = dddd.window.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=dddd.width//2, y=dddd.height//2,
                          anchor_x='center', anchor_y='center')
#label.draw()
lab = MyLabel()
dddd.add_onetime(label)
label.draw()
ret = dddd.render(return_rgb_array = mode=='rgb_array')

        

window = pyglet.window.Window()
label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')

@window.event
def on_draw():
    #window.clear()
    label.draw()

pyglet.app.run()
