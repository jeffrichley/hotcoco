import gym
from gym import spaces
from gym.envs.classic_control import rendering
import math
import numpy as np




class HotEnv(gym.Env):

    def __init__(self):

        self.fish_start_state = (100, 100, 0.)  # fish starts in the middle pointing up
        self.shark_1_start_state = (90, 150, math.pi)  # fish starts at the top pointing down
        self.shark_2_start_state = (110, 150, math.pi)  # fish starts at the top pointing down

        self.fish_state = self.fish_start_state  # fish starts in the middle pointing up
        self.shark_1_state = self.shark_1_start_state  # fish starts at the top pointing down
        self.shark_2_state = self.shark_2_start_state  # fish starts at the top pointing down

        self.state = [0]
        self.step_count = 60

        self.action_space = spaces.Discrete(11)


        # rendering information
        self.render_scale = 3
        self.fish_sight_distance = 15
        self.fish_flee_distance = 10
        self.fish_sight_spread_angle = 60
        self.fish_sight_spread_radians = self.fish_sight_spread_angle * math.pi / 180
        self.shark_swim_distance = 10
        self.shark_eat_distance = 4


        # rendering pieces
        self.viewer = None
        self.fish_transform = None
        self.shark_1_transform = None
        self.shark_2_transform = None

    def get_updated_state(self, state, rotation):
        x, y, r = state
        x -= self.shark_swim_distance * math.sin(rotation + r)
        y += self.shark_swim_distance * math.cos(rotation + r)

        return x, y, r + rotation

    def in_zone(self, shark_state, shark_distance):

        # we need to be at least close enough for the fish to see
        if shark_distance <= self.fish_sight_distance * self.render_scale:
            delta_x = self.fish_state[0] - shark_state[0]
            delta_y = self.fish_state[1] - shark_state[1]
            shark_angle = math.atan2(delta_y, delta_x) + 1.5708  # needed to add 90 degrees for correct orientation
            neg_shark_angle = shark_angle - (2 * math.pi)

            fish_angle = self.fish_state[2]

            # check to see if it is in the viewing angle
            # neg angle is used for when the one angle went negative but the calculated angle is still positive
            if (shark_angle <= fish_angle + self.fish_sight_spread_radians and shark_angle >= fish_angle - self.fish_sight_spread_radians) or \
               (neg_shark_angle <= fish_angle + self.fish_sight_spread_radians and neg_shark_angle >= fish_angle - self.fish_sight_spread_radians):
                return True

        return False

    def step(self, action):
        self.step_count -= 1

        done = self.step_count <= 0

        # by default the agent gets -1 for each time step
        reward = -1

        shark_1_rotation, shark_2_rotation = action
        shark_1_rotation = (shark_1_rotation - ((self.action_space.n - 1) / 2)) * 0.0872665  # 5 degrees
        shark_2_rotation = (shark_2_rotation - ((self.action_space.n - 1) / 2)) * 0.0872665  # 5 degrees

        self.shark_1_state = self.get_updated_state(self.shark_1_state, shark_1_rotation)
        self.shark_2_state = self.get_updated_state(self.shark_2_state, shark_2_rotation)

        shark_1_distance = math.sqrt((self.shark_1_state[0] - self.fish_state[0])**2 + (self.shark_1_state[1] - self.fish_state[1])**2)
        shark_2_distance = math.sqrt((self.shark_2_state[0] - self.fish_state[0]) ** 2 + (self.shark_2_state[1] - self.fish_state[1]) ** 2)

        shark_1_in_zone = self.in_zone(self.shark_1_state, shark_1_distance)
        shark_2_in_zone = self.in_zone(self.shark_2_state, shark_2_distance)

        point = None
        shark_distance = math.inf
        if (shark_1_distance < shark_2_distance and shark_1_distance <= self.fish_sight_distance * self.render_scale and shark_1_in_zone) or \
                (not shark_2_in_zone and shark_1_distance <= self.fish_sight_distance * self.render_scale and shark_1_in_zone):
            point = self.shark_1_state[0], self.shark_1_state[1]
            shark_distance = shark_1_distance
        elif (shark_2_distance < shark_1_distance and shark_2_distance <= self.fish_sight_distance * self.render_scale and shark_2_in_zone) or \
                (not shark_1_in_zone and shark_2_distance <= self.fish_sight_distance * self.render_scale and shark_2_in_zone):
            point = self.shark_2_state[0], self.shark_2_state[1]
            shark_distance = shark_2_distance

        x, y, r = self.fish_state

        if point is not None:
            # update the fish's looking direction
            delta_x = x - point[0]
            delta_y = y - point[1]
            r = math.atan2(delta_y, delta_x) + 1.5708  # needed to add 90 degrees for correct orientation

            # if the shark got too close, we are done and the fish runs away
            if shark_distance <= self.fish_flee_distance * self.render_scale:
                done = True
                reward = -100

        if shark_1_distance <= self.shark_eat_distance * self.render_scale or shark_2_distance <= self.shark_eat_distance * self.render_scale:
            done = True
            reward = 100

        self.fish_state = (x, y, r)

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.fish_state = self.fish_start_state  # fish starts in the middle pointing up
        self.shark_1_state = self.shark_1_start_state  # fish starts at the top pointing down
        self.shark_2_state = self.shark_2_start_state  # fish starts at the top pointing down
        self.step_count = 60

    def draw_fan(self, transform, distance, red, blue, green):
        prev_angle = -self.fish_sight_spread_angle
        for angle in range(-self.fish_sight_spread_angle, self.fish_sight_spread_angle + 1, 5):
            if angle != prev_angle:
                first_x = distance * self.render_scale * math.sin(prev_angle * math.pi * 2 / 360) * self.render_scale
                first_y = distance * self.render_scale * math.cos(prev_angle * math.pi * 2 / 360) * self.render_scale
                second_x = distance * self.render_scale * math.sin(angle * math.pi * 2 / 360) * self.render_scale
                second_y = distance * self.render_scale * math.cos(angle * math.pi * 2 / 360) * self.render_scale
                fish_sight = rendering.FilledPolygon([(0, 0), (first_x, first_y), (second_x, second_y)])
                fish_sight.set_color(red, blue, green)
                self.viewer.add_geom(fish_sight)
                fish_sight.add_attr(transform)
            prev_angle = angle

    def add_shark(self, shark_1_transform):
        shark_1 = rendering.make_circle()
        # shark_1_transform = rendering.Transform()
        shark_1.add_attr(shark_1_transform)
        shark_1.set_color(0.0588, 0.6156, 0.3450)
        shark_1_pointing = rendering.Line((0, 0), (0, 17))
        shark_1_pointing.set_color(0.0588, 0.6156, 0.3450)
        shark_1_pointing.add_attr(shark_1_transform)
        self.viewer.add_geom(shark_1_pointing)
        self.viewer.add_geom(shark_1)

    def render(self, mode='human'):

        if self.viewer is None:
            screen_width = 200 * self.render_scale
            screen_height = 200 * self.render_scale

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # make the fish
            fish = rendering.make_circle()
            self.fish_transform = rendering.Transform()
            fish.add_attr(self.fish_transform)
            fish.set_color(0.2588, 0.5215, 0.9568)

            # make the fish visibility area
            self.draw_fan(self.fish_transform, self.fish_sight_distance, 0.9568, 0.6274, 0)

            # make the fish flee area
            self.draw_fan(self.fish_transform, self.fish_flee_distance, 0.8588, 0.2666, 0.2156)

            # add the fish after the vision parts so it is on top
            self.viewer.add_geom(fish)

            # add shark 1
            self.shark_1_transform = rendering.Transform()
            self.add_shark(self.shark_1_transform)

            # add shark 2
            self.shark_2_transform = rendering.Transform()
            self.add_shark(self.shark_2_transform)

        # perform the translations
        self.shark_1_transform.set_translation(self.shark_1_state[0] * self.render_scale, self.shark_1_state[1] * self.render_scale)
        self.shark_2_transform.set_translation(self.shark_2_state[0] * self.render_scale,self.shark_2_state[1] * self.render_scale)
        self.fish_transform.set_translation(self.fish_state[0] * self.render_scale, self.fish_state[1] * self.render_scale)

        # perform the rotations
        self.shark_1_transform.set_rotation(self.shark_1_state[2])
        self.shark_2_transform.set_rotation(self.shark_2_state[2])
        self.fish_transform.set_rotation(self.fish_state[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')