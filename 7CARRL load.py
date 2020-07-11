
import numpy as np
import pygame
from keras.models import Sequential, load_model
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

'''DQN presets'''

filepath = "models/3CARRL-1590830970-50-0.9521577859830145-.h5"
#add filepath of saved model to load it

DISCOUNT = 0.95
REPLAY_MEMORY_SIZE = 50_000 #how many last steps to keep for training
MIN_REPLAY_MEMORY_SIZE = 1000 #minimum number of steps to start training
MINIBATCH_SIZE = 64 #how many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5 #terminal states (end of episodes)
MODEL_NAME = "7CARRL"
MIN_REWARD = 0
MEMORY_FRACTION = 0.20

#environment settings
EPISODES = 10_000
hyphen_n = 0
previous_ep = ""
epsilon = ""

for char in filepath:
    if char == "-":
        hyphen_n += 1
    elif hyphen_n == 2:
        previous_ep += char
    elif hyphen_n == 3:
        epsilon += char

previous_ep = int(previous_ep) #current episode

epsilon = float(epsilon)

#exploration settings
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.1

#stats settings
AGGREGATE_STATS_EVERY = 20 #episodes
SAVE_MODEL_EVERY = 50
SHOW_PREVIEW = True

'''end of DQN presets'''

'''car env colors'''
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

'''end of car env colors'''

'''car env elements'''

#different routes for the car env
#[startpt, endpt, road_center_pts, road_rects]

routes = { "diagup": [(270, 345), (425, 185), [(270, 345), (425, 185)], [[(248.21952435952647, 323.8508082547879), (402.6860622731231, 164.40147879559137), (445.7804756404735, 206.1491917452121), (291.3139377268769, 365.5985212044086)]]],
           "diagup forward": [(274, 341), (456, 246), [(274, 341), (346, 246), (455, 243)], [[(250.45377448177322, 321.90064059034813), (322.33205853927774, 227.06123801447416), (370.1502447119873, 263.3023896401067), (298.2719606544828, 358.1417922159807)], [(345.1950656945742, 215.49703215842314), (454.15380473118125, 212.49816778126882), (455.80455576447724, 272.47545532435527), (346.84581672787016, 275.47431970150956)]]],
           "diagup diagdown": [(254, 346), (508, 346), [(254, 346), (380, 261), (507, 345)], [[(238.04750551478529, 320.0734769324277), (363.2266855210932, 235.6272047059501), (396.78149567730947, 285.3672762316354), (271.6023156710015, 369.81354845811296)], [(396.160981669893, 236.05146482654084), (522.9389415471204, 319.9046036429747), (489.839018330107, 369.94853517345916), (363.0610584528796, 286.0953963570253)]]],
           "forward diagup": [(242, 320), (480, 214), [(242, 320), (409, 313), (480, 213)], [[(240.81643630719975, 289.5023094594974), (407.6699227565286, 282.50845074605246), (410.18268636614954, 342.45581114700894), (243.32919991682067, 349.44966986045387)], [(384.2242733974875, 295.370704489425), (454.8527475252959, 195.8939803657512), (503.7757266025125, 230.629295510575), (433.1472524747041, 330.1060196342488)]]],
           "forward": [(268, 283), (481, 284), [(268, 283), (481, 284)], [[(268.14201169746957, 252.50268351901553), (481.1396643184339, 253.50267249845666), (480.8579772819716, 313.5020112649255), (267.86032466100727, 312.5020222854844)]]],
           "forward diagdown": [(235, 292), (491, 402), [(235, 292), (403, 297), (489, 400)], [[(235.9296396564792, 261.51438422320416), (403.8552842873008, 266.51217126578814), (402.0703603435208, 326.48561577679584), (234.14471571269922, 321.48782873421186)], [(426.08685340317015, 277.34261119590417), (511.9697506269229, 380.2023601964453), (465.91314659682985, 418.65738880409583), (380.0302493730771, 315.7976398035547)]]],
           "diagdown diagup": [(214, 220), (457, 223), [(214, 220), (335, 307), (457, 220)], [[(231.43144628988748, 195.44322213849105), (352.4069278219837, 282.42559315743625), (317.3804696935494, 331.14055216364955), (196.40498816145316, 244.15818114470432)], [(318.33217119419504, 281.5393734681669), (439.64560924471255, 195.0289709239454), (474.482012953795, 243.8800198033484), (353.1685749032775, 330.39042234756994)]]],
           "diagdown forward": [(204, 206), (450, 336), [(204, 206), (299, 330), (449, 335)], [[(227.37767971136145, 187.83775212619736), (322.25106363483656, 311.6724848263122), (274.62232028863855, 348.16224787380264), (179.74893636516344, 224.3275151736878)], [(300.0410768835426, 299.5180405231775), (449.95781293041574, 304.5152650580733), (447.9589231164574, 364.4819594768225), (298.04218706958426, 359.4847349419267)]]],
           "diagdown": [(222, 212), (425, 352), [(222, 212), (425, 352)], [[(238.7768160065271, 187.47246285869068), (441.28716134169247, 327.13476998639095), (407.2231839934729, 376.5275371413093), (204.71283865830753, 236.86523001360905)]]] }

car_img = pygame.image.load("pixelcar.png")

def update_quit(): #at the start of every loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

def draw_pts_and_road(): #draw the screen, start and end points, and race track
    global bg_color, startpt, endpt, road
    env.screen.fill(bg_color)
    env.road.draw(env.screen)
    env.startpt.draw(env.screen)
    env.endpt.draw(env.screen)

class Car:
    def __init__(self, x, y):
        #blue rectangular car
        
        def round_to_five(deg):
            #rounds the angle in deg to nearest 5 (for the car to be blit in straight lines)
            return 5 * round(deg/5)

        self.rotation = -round_to_five(math.degrees(math.atan2((env.road.road_center_pts[1][1]-y), (env.road.road_center_pts[1][0]-x))))
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

class Checkpoint:
    def __init__(self, x, y, colour):
        self.rect = pygame.Rect(0, 0, 30, 30)
        self.rect.centerx = x
        self.rect.centery = y
        self.colour = colour

    def draw(self, screen):
        pygame.draw.rect(screen, self.colour, self.rect)

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

def get_input_array():
    #128x128 image centered on racecar
    square = pygame.Rect(0,0,128,128)
    square.center = env.racecar.rect.center
    square = env.screen.subsurface(square) #128x128 surface with racecar in it

    input_array = pygame.surfarray.array3d(square)
    #rgb array of 128x128 image

    return np.array(input_array)

'''end of car env elements'''

'''car environment'''
class CarEnv:
    MOVE_PENALTY = 3
    LONG_STOP_PENALTY = 200 #penalty if car just rotates and stop moving for a long time
    COLLIDE_PENALTY = 300
    ENDPT_REWARD = 200
    NRCP_REWARD = 20 #reward for moving closer to the next road centrepoint
    OBSERVATION_SPACE_VALUES = (128, 128, 3) #128x128 rgb image
    ACTION_SPACE_SIZE = 5 #5 possible actions
    MAX_EP_STEPS = 300 #maximum number of steps that car can take in an episode
    INPUT_COLORS = 255 #rgb colors

    screen_width = 900
    screen_height = 600
    size = (screen_width, screen_height)

    def reset(self): #reset variables that changed during each episode
        self.episode_step = 0

        self.eps_car_stopped = 0 #no. of consecutive episodes car stops moving and just rotates
        
        self.screen = pygame.display.set_mode(self.size) #reset screen surface

        #the preset race track route
        self.route_name, self.car_route = random.choice(list(routes.items()))
        
        self.startpt = Checkpoint(self.car_route[0][0], self.car_route[0][1], startpt_color)
        self.endpt = Checkpoint(self.car_route[1][0], self.car_route[1][1], endpt_color)
        self.road = Road([])
        self.road.road_center_pts = self.car_route[2]
        self.road.road_rects = self.car_route[3]

        draw_pts_and_road()
        
        self.racecar = Car(self.startpt.rect.centerx, self.startpt.rect.centery)
        self.racecar.draw(self.screen)

        self.outside_track_mask = pygame.mask.from_threshold(self.screen, bg_color, (5,5,5,255))

        self.ncpt_index = 1
        self.ncpt = self.road.road_center_pts[self.ncpt_index]
        #next road centerpoint from car (skip index 0 as it is already the startpt)

        self.dist_to_ncpt = ((self.ncpt[0]-self.racecar.rect.centerx)**2 + (self.ncpt[1]-self.racecar.rect.centery)**2)**(0.5)
        #distance from car to next road centerpoint
        self.prev_recorded_dist = self.dist_to_ncpt
        #prev recorded distance from car to next road centerpoint
        
        observation = get_input_array()
        #gets 128x128 image array containing the racecar

        return observation

    def step(self, action):
        update_quit() #in case pygame quits
        
        self.episode_step += 1
        self.racecar.update(action) #to move the car (speed and direction)

        draw_pts_and_road()
        self.racecar.draw(self.screen)

        new_observation = get_input_array()

        if self.racecar.speed == 0: #if car stops and just rotates on the spot
            self.eps_car_stopped += 1
        else:
            self.eps_car_stopped = 0

        racecar_mask = pygame.mask.from_surface(self.racecar.image) #mask of the racecar
        #for collision detection

        self.dist_to_ncpt = ((self.ncpt[0]-self.racecar.rect.centerx)**2 + (self.ncpt[1]-self.racecar.rect.centery)**2)**(0.5)
        #distance from car to next road centerpoint
        
        if self.dist_to_ncpt < self.road.width//2 and (self.ncpt_index+1) != len(self.road.road_center_pts):
            #getting close to the current road centerpoint
            self.ncpt_index += 1
            self.ncpt = self.road.road_center_pts[self.ncpt_index]
            #change to the next road centerpoint
            
            self.dist_to_ncpt = ((self.ncpt[0]-self.racecar.rect.centerx)**2 + (self.ncpt[1]-self.racecar.rect.centery)**2)**(0.5)
            self.prev_recorded_dist = self.dist_to_ncpt
            
        done = False

        '''episode outcomes and rewards'''
        if self.outside_track_mask.overlap_area(racecar_mask, self.racecar.rect.topleft) > 0:
            #if car collides with area outside track
            reward = -self.COLLIDE_PENALTY
            done = True
            print("COLLIDED")

        elif ((self.endpt.rect.centerx-self.racecar.rect.centerx)**2 + (self.endpt.rect.centery-self.racecar.rect.centery)**2)**(0.5) < 40:
            #if car hits the endpoint
            reward = self.ENDPT_REWARD
            done = True
            print("ENDPT")

        elif (self.prev_recorded_dist - self.dist_to_ncpt) >= 5:
            #if car is getting closer to next road centerpoint
            reward = self.NRCP_REWARD
            self.prev_recorded_dist = self.dist_to_ncpt
            
        elif self.eps_car_stopped >= 20:
            #if car just stops and rotates on the spot for a long time
            reward = -self.LONG_STOP_PENALTY
            done = True
            print("LONGSTOP")
        
        else:
            #if car moves and isn't getting closer to next road centrepoint by 5 pixels
            reward = -self.MOVE_PENALTY

        if self.episode_step >= self.MAX_EP_STEPS:
            #if number of steps in the episode reaches the max value
            done = True
            print("OUTOFSTEPS")

        pygame.display.flip()

        clock.tick(60)

        return new_observation, reward, done

    def render(self):
        pygame.display.flip()

'''end of car environment'''

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

        #target model that we predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        #array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        #custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}_{int(time.time())}")

        #to count when to update target model with main model's weights
        self.target_update_counter = 0

    def create_model(self):
        '''
        input: 128x128 image array of current state
        2 convolution layers and 2 dense layers
        output: q values of all next possible actions to take
        '''
        model = Sequential()

        #first convolution layer
        model.add(Conv2D(128, (3,3), input_shape=env.OBSERVATION_SPACE_VALUES))
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
        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear")) #output layer

        model.summary() #print model architecture
        
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        #add step's data to a memory replay array
        #(observation space, action, reward, new observation space, done)
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        #train main model every step during episode

        #start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        #get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        #input current states of minibatch into main model to get q values
        current_states = np.array([transition[0] for transition in minibatch])/env.INPUT_COLORS
        current_qs_list = self.model.predict(current_states)
        print("curr: ", current_qs_list)

        #input future states of minibatch into target model to get q values
        #weights of target model more stable and less random than main model
        new_current_states = np.array([transition[3] for transition in minibatch])/env.INPUT_COLORS
        future_qs_list = self.target_model.predict(new_current_states)
        print("future: ", future_qs_list)

        X = [] #images of current state
        y = [] #new updated q values of current state after taking the next best possible action

        #enumerate batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            if not done: #not terminal state, get new q values of future states
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q

            else: #terminal state (done)
                new_q = reward

            #update q value of given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            #append to training data
            X.append(current_state)
            y.append(current_qs)

        #fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/env.INPUT_COLORS, np.array(y), batch_size=MINIBATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=[self.tensorboard]
                       if terminal_state else None)

        #update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        #if counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        #predicts q values when you input current state into the main model
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/env.INPUT_COLORS)[0]


'''main code'''

pygame.init() #initialise pygame
pygame.display.set_caption("Self Driving Car")
clock = pygame.time.Clock()

np.set_printoptions(threshold=sys.maxsize) #allow full arrays to be printed

env = CarEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when training multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

#deep q network agent
agent = DQNAgent()

new_ep = 0 #number of new episodes

#iterate over episodes
for episode in tqdm(range(previous_ep, EPISODES + 1), ascii=True, unit='episodes'):

    #update tensorflow step every episode
    agent.tensorboard.step = episode

    #add one to the number of new episodes
    new_ep += 1

    #restarting episode- reset episode reward and step number
    episode_reward = 0
    step = 1

    #reset environment and get initial state
    current_state = env.reset()

    #reset flag and start iterating until episode ends
    done = False
    while not done:

        #epsilon adds randomness
        if np.random.random() > epsilon:
            #get next best possible action from q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            #get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        #transform new continuous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        #every step, update replay memory and train main model
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    #append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    print("ROUTE: ", env.route_name)
    print("REWARD: ", episode_reward)
    
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                       reward_max=max_reward, epsilon=epsilon)
        print("AVG REWARD: ", average_reward)
        
        #save model only when minimum reward is greater or equal a set value
        if episode == 1 or min_reward >= MIN_REWARD:
            agent.model.save(f"models/{MODEL_NAME}-{int(time.time())}-{episode}-{epsilon}-avgreward{average_reward}.h5")
        
    if not new_ep % SAVE_MODEL_EVERY:
        agent.model.save(f"models/{MODEL_NAME}-{int(time.time())}-{new_ep+previous_ep}-{epsilon}-.h5")
        print("Model saved at ", str(episode), "episodes.")
        
    #decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

        
