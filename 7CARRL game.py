
import numpy as np
import pygame
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
from collections import deque
import time
import math
import random
from tqdm import tqdm
import os, sys

filepath = "models/7CARRL-1593485710-1300-0.4996543604022426UStg-.h5"
#add filepath of saved model to load it

pygame.init()
np.set_printoptions(threshold=sys.maxsize)

screen_width = 900
screen_height = 600
size = (screen_width, screen_height)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Self Driving Car")
clock = pygame.time.Clock()

'''colors'''
bg_color = [30, 30, 30]
bg_int = 505290240
bg_input = 0 #integer to represent the background pixels when inputting into the neural network

road_color = [230, 230, 230]
road_int = -421075456
road_input = 2

startpt_color = [65, 199, 2]
startpt_int = 46612736
startpt_input = 1

endpt_color = [180, 0, 0]
endpt_int = 46080
endpt_input = 5

#car colors
carblack = 16843008 #black outline
darkblue1 = -1302589952 #variations of dark blue color of car body
darkblue2 = -1302655488
darkblue3 = -1285812736
carbody_input = 3

lightblue1 = -37326592 #light blue windshield
lightblue2 = -37326848
carwhite1 = -256 #white headlight
carwhite2 = 256
carfront_input = 4

'''end of colors'''

arial_80_font = pygame.font.SysFont("Arial", 80)
arial_40_font = pygame.font.SysFont("Arial", 40)
car_img = pygame.image.load("pixelcar.png")

def update_quit(): #at the start of every loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

def draw_pts_and_road(): #draw the screen, start and end points, and race track
    global screen, bg_color, startpt, endpt, road
    screen.fill(bg_color)
    road.draw(screen)
    startpt.draw(screen)
    endpt.draw(screen)

def update_display(): #at the end of every loop
    global clock
    pygame.display.flip()
    clock.tick(60)

def centred_rect(rect_width, rect_height, rect_top_y):
    return pygame.Rect(screen_width//2 - rect_width//2, rect_top_y, rect_width, rect_height)

class Car:
    def __init__(self, x, y):
        #blue rectangular car

        def round_to_five(deg):
            #rounds the angle in deg to nearest 5 (for the car to be blit in straight lines)
            return 5 * round(deg/5)

        self.rotation = -round_to_five(math.degrees(math.atan2((road.road_center_pts[1][1]-y), (road.road_center_pts[1][0]-x))))
        #the car faces the direction of the first road center point
        
        self.image = pygame.transform.rotate(car_img, self.rotation)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y

        self.centerx_float = x
        self.centery_float = y

        self.speed = 0

    def draw(self, screen):
        screen.blit(self.image, self.rect)

        '''
        #to display a 128x128 box around the racecar (width of box is 2)
        square = pygame.Rect(0,0,130,130)
        square.center = self.rect.center
        pygame.draw.rect(screen, [255,0,0], square, 2)
        '''

    def update(self, action):
        global startpt, road, win, no_collision, outside_track_mask

        #5 possible actions to take
        if action == 0:
            self.rotation += 5 #rotate anticlockwise
            self.speed = 5

        elif action == 1:
            self.rotation += 5
            self.speed = 0

        elif action == 2:
            self.rotation -= 5 #rotate clockwise
            self.speed = 5

        elif action == 3:
            self.rotation -= 5
            self.speed = 0

        elif action == 4:
            self.rotation += 0 #no change in rotation
            self.speed = 5

        else:
            print("ERROR: No choice of action given.")

        orig_centerx = self.rect.centerx
        orig_centery = self.rect.centery

        self.image = pygame.transform.rotate(car_img, self.rotation)
        self.rect = self.image.get_rect()
        self.rect.centerx = orig_centerx
        self.rect.centery = orig_centery
        
        if self.speed > 0:
            #storing centerx value as a float because rounding off to a integer (in self.rect.centerx) is inaccurate
            self.centerx_float += self.speed * math.cos(math.radians(self.rotation))
            self.centery_float -= self.speed * math.sin(math.radians(self.rotation))
            #change in y is negative because the y coords decrease as you go up the screen

            self.rect.centerx = self.centerx_float
            self.rect.centery = self.centery_float

        #collision detection
        racecar_mask = pygame.mask.from_surface(self.image) #mask of the racecar

        #detect collision with outside track
        if outside_track_mask.overlap_area(racecar_mask, self.rect.topleft) > 0:
            no_collision = False

        if ((endpt.rect.centerx-racecar.rect.centerx)**2 + (endpt.rect.centery-racecar.rect.centery)**2)**(0.5) < 40:
            #if car hits the endpoint
            win = True

class Checkpoint:
    def __init__(self, x, y, colour):
        self.rect = pygame.Rect(0, 0, 30, 30)
        self.rect.centerx = x
        self.rect.centery = y
        self.colour = colour

    def draw(self, screen):
        pygame.draw.rect(screen, self.colour, self.rect)

def rotate_rect(rect, angle, centerpt, width, length):
    #takes in rect and angle (rad) and rotates the rect about a center point (gives coords of the 4 vertices)
    
    def rotated_coord(x, y, centerptx, centerpty, angle):
        newx = (x-centerptx)*math.cos(angle) - (y-centerpty)*math.sin(angle) + centerptx
        newy = (x-centerptx)*math.sin(angle) + (y-centerpty)*math.cos(angle) + centerpty
        return (newx, newy)

    topleft = rotated_coord(rect.left, rect.top, centerpt[0], centerpt[1], angle)
    topright = rotated_coord(rect.right, rect.top, centerpt[0], centerpt[1], angle)
    botright = rotated_coord(rect.right, rect.bottom, centerpt[0], centerpt[1], angle)
    botleft = rotated_coord(rect.left, rect.bottom, centerpt[0], centerpt[1], angle)
    
    return [topleft, topright, botright, botleft]

class Road:
    def __init__(self, road_center_pts):
        self.width = 60
        self.road_center_pts = road_center_pts
        #list of tuples with the x and y coords of road centre points

        self.road_rects = []
        #list containing lists of the 4 vertices of each road segment

    def draw(self, screen):

        for road_rect in self.road_rects:
            pygame.draw.polygon(screen, road_color, road_rect)

        for coord in self.road_center_pts:
            pygame.draw.circle(screen, road_color, coord, self.width//2)

    def add_new_point(self, newpoint):
        prevpt = self.road_center_pts[-2]
        road_length = ((newpoint[0]-prevpt[0])**2 + (newpoint[1]-prevpt[1])**2)**(0.5)
        #road length is distance between the current point and previous point
        new_rect = pygame.Rect(0, 0, road_length, self.width)
                
        new_rect.center = ((newpoint[0]+prevpt[0])/2, (newpoint[1]+prevpt[1])/2)
        #midpoint between the current point and previous point

        angle = math.atan2((newpoint[1]-prevpt[1]), (newpoint[0]-prevpt[0]))
        self.road_rects.append(rotate_rect(new_rect, angle, new_rect.center, self.width, road_length))
        #add a list of the 4 vertices of the new diagonal road rectangle

def get_input_array():
    #128x128 image centered on racecar
    square = pygame.Rect(0,0,128,128)
    square.center = racecar.rect.center
    square = screen.subsurface(square) #128x128 surface with racecar in it

    input_array = pygame.surfarray.array3d(square)
    #rgb array of 128x128 image
    
    return np.array(input_array)

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._log_write_dir = self.log_dir
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass
    
    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(f'{key}', value, step=self.step)

            self.writer.flush()

class DQNAgent:
    def __init__(self):
        #main model that is trained every step
        self.model = load_model(filepath)

        self.OBSERVATION_SPACE_VALUES = (128, 128, 3)
        self.ACTION_SPACE_SIZE = 5

    def create_model(self):
        '''
        input: 128x128 image array of current state
        2 convolution layers and 2 dense layers
        output: q values of all next possible actions to take
        '''
        model = Sequential()

        #first convolution layer
        model.add(Conv2D(128, (3,3), input_shape=self.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        #second convolution layer
        model.add(Conv2D(128, (3,3))) #input is already defined by the output of the previous layer
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        #dense layer
        model.add(Flatten())
        model.add(Dense(64))

        #dense output layer
        model.add(Dense(self.ACTION_SPACE_SIZE, activation="linear")) #output layer

        model.summary() #print model architecture
        
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def get_qs(self, state):
        #predicts q values when you input current state into the main model
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

#main code

agent = DQNAgent()
                
'''start screen'''
start_screen = True

while start_screen:
    update_quit()
    
    screen.fill([10,10,10])
        
    #start button
    start_button = centred_rect(300, 100, 400)
    pygame.draw.rect(screen, [255,255,255], start_button)
    start_text = arial_80_font.render("START", True, [0,0,0])
    screen.blit(start_text, centred_rect(180, 100, 425))

    if pygame.mouse.get_pressed()[0] and start_button.collidepoint(pygame.mouse.get_pos()):
        #if the cursor clicks on the start button, game starts
        start_screen = False
        time.sleep(0.2)

    update_display()

'''end of start screen'''

game_playing = True

turn_count = 0 #prevent car from turning for a long time

while game_playing:
    '''set start and end points'''
    no_startpt = True
    no_endpt = True
    wait_for_restart = True

    while no_startpt: #set startpt
        update_quit()
        screen.fill(bg_color)
        
        if pygame.mouse.get_pressed()[0]: #create startpoint at the position of cursor on click
            startpt = Checkpoint(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1], startpt_color)
            time.sleep(0.2) #prevent repeated detection of clicks
            no_startpt = False
            print("startpt set.")

        update_display()

    while no_endpt: #set endpt
        update_quit()
        screen.fill(bg_color)
        startpt.draw(screen)

        if pygame.mouse.get_pressed()[0]: #create endpoint at the position of cursor on click
            endpt = Checkpoint(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1], endpt_color)
            time.sleep(0.2) #prevent repeated detection of clicks
            no_endpt = False
            print("endpt set.")

        update_display()

    '''end of set start and end points'''

    '''draw race track'''
    no_full_road = True

    road = Road([(startpt.rect.centerx, startpt.rect.centery)])
    #add first road center point

    while no_full_road: #draw race track
        update_quit()
        screen.fill(bg_color)
        if len(road.road_center_pts) > 1:
            road.draw(screen)
        startpt.draw(screen)
        endpt.draw(screen)

        if pygame.mouse.get_pressed()[0]: #add new road center point
            road.road_center_pts.append((pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]))
            road.add_new_point(road.road_center_pts[-1])
            time.sleep(0.2)

            if endpt.rect.collidepoint(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]):
                #if the road drawn reaches the endpoint
                no_full_road = False

        update_display()

    '''end of draw race track'''

    '''racecar controls'''
    racecar = Car(startpt.rect.centerx, startpt.rect.centery)

    draw_pts_and_road()
    update_display()

    outside_track_mask = pygame.mask.from_threshold(screen, bg_color, (5,5,5,255))
    #creates mask of the area outside the race track (for collision detection)

    no_collision = True
    win = False

    while no_collision:
        update_quit()
        draw_pts_and_road() #draw the screen, start and end points, and race track

        action = np.argmax(agent.get_qs(get_input_array()))
        #get next best action from neural network

        if action == 1 or action == 3:
            turn_count += 1
        else:
            turn_count = 0

        if turn_count == 20:
            action = random.choice([0,2,4])
            turn_count = 0
        
        racecar.update(action)
        racecar.draw(screen)

        #restart button
        restart_button = pygame.Rect(710, 520, 170, 60)
        pygame.draw.rect(screen, [255,255,255], restart_button)
        start_text = arial_40_font.render("RESTART", True, [0,0,0])
        screen.blit(start_text, pygame.Rect(732, 535, 90, 50))

        if pygame.mouse.get_pressed()[0] and restart_button.collidepoint(pygame.mouse.get_pos()):
            #if the cursor clicks on the start button, game starts
            no_collision = False
            wait_for_restart = False
            time.sleep(0.2)

        if win:
            break

        update_display()

    '''end of racecar controls'''

    if win:
        print("It won!")
    else:
        print("It lost.")


    while wait_for_restart:
        update_quit()
        
        draw_pts_and_road()
        racecar.draw(screen)
        
        #restart button
        restart_button = pygame.Rect(710, 520, 170, 60)
        pygame.draw.rect(screen, [255,255,255], restart_button)
        start_text = arial_40_font.render("RESTART", True, [0,0,0])
        screen.blit(start_text, pygame.Rect(732, 535, 90, 50))

        update_display()

        if pygame.mouse.get_pressed()[0] and restart_button.collidepoint(pygame.mouse.get_pos()):
            #if the cursor clicks on the start button, game starts
            wait_for_restart = False
            time.sleep(0.2)

#pygame.quit()

