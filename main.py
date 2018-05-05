import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image
#import matplotlib.image as mpimg

const_card_hight    = 100
const_card_width    = 100
const_RGB_red       = [1,0,0]    # colouring = red
const_RGB_yellow    = [1,1,0]    # colouring = red
right, up, left, down   = 0, 1, 2, 3

class Map:
    """This is the class of the map depicted as occupancy grid. """
    def __init__(self):
        self.grid = np.zeros((const_card_hight, const_card_width))

# class ProbabilityMap(Map):
#     def __init__(self, initial_belief_grid):
#         self.grid = initial_belief_grid
#     def showMap(self):
#         plt.imshow(self.grid)
#         plt.show()

class EnvironmentMap(Map):
    def __init__(self, occupancy_grid):
        self.grid = occupancy_grid
        self.dist_grid = self._createDistanceGrid()

    def _createDistanceGrid(self):
        distance_grid = np.zeros((const_card_hight, const_card_width, 4))
        """update the distances in the direction "right" and "left" of the cells in grid[row,col_prev:col"""
        for row in range(0, const_card_hight):     # loop through rows
            col_found = False
            for col in range(0, const_card_width): # loop through columns
                if self.grid[row,col] == 0:  #if the cell IS occupied --> calculate the distances for each horizontal direction

                    # if another occupied cell is found in the current row
                    # --> calculate the left and/or right distances for each cell
                    if col_found == True:
                        for update_col in range(prev_col+1, col):
                            distance_grid[row, update_col, right]   = col - update_col -1       #calculate distance to right object
                            distance_grid[row, update_col, left]    = update_col - prev_col -1  #calculate distance to left object
                            prev_col = col  # set right object to prev_object and continue looping from left to right
                    else:
                        prev_col = col  #first occupied cell found (in the current row)

                    if col_found == False:
                        col_found = True    #at least one occupied cell found


        """update the distances in the vertical directions "up" and "down" of the cells in grid[prev_row:row,col]"""
        for col in range(0, const_card_width):     # loop through columns
            row_found = False   # occupied row in current column not found in the beginning
            for row in range(0, const_card_hight): # loop through rows
                if self.grid[row, col] == 0:  #if the cell IS occupied --> calculate the distances for each vertical direction

                    # if another occupied cell is found in the current column
                    # --> calculate the up and/or down distances for each cell
                    if row_found == True:
                        for update_row in range(prev_row+1, row):
                            distance_grid[row, update_row, up]      = row - update_row -1       #calculate distance to right object
                            distance_grid[row, update_row, down]    = update_row - prev_row -1  #calculate distance to left object
                            prev_row = row  # set current bottom object to prev_object and continue looping from top to bottom
                    else:
                        prev_row = row  #first occupied cell found (in the current column)

                    if row_found == False:
                        row_found = True    #at least one occupied cell found
        return distance_grid

    def showMap(self):
        plt.imshow(self.grid)
        plt.show()

class Robot:
    """ This is the class of the robot:
        The roboter has a position and a direction
    """
    i = 0
    def __init__(self, initial_belief_grid, environment_map, measured_distances):
        self.pos_belief = initial_belief_grid
        self.environment = environment_map
        self.direction = right
        self.measured_distance = measured_distances

    def move(self, direction):
        self.direction = direction
        prior_belief = _calcPriorBelief()
        posterior_belief =

    # in reality this function would provide the measured distance values,
    # for testing purpose this measurements are hard coded
    def _getDistance(self):
        ++i
        return self.measured_distance[i-1]

    def _calcPosteriorBelief(self):
        measured_distance = self._getDistance()
        distance_belief_grid = np.zeros((const_card_hight, const_card_width))
        # for every cell calculate the possibility according to the measured distance:
        for row in range(0, const_card_hight):     # loop through rows
            for col in range(0, const_card_width): # loop through columns
                abberation = abs(measured_distance - self.environment.distance_grid[row, col, self.direction])
                if  abberation == 0:
                    distance_belief_grid[row, col] = 0.4
                if  abberation == 1:
                    distance_belief_grid[row, col] = 0.2
                if  abberation == 2:
                    distance_belief_grid[row, col] = 0.1
        return distance_belief_grid

    def _calcPriorBelief(self):
        prior_belief_grid = np.zeros((const_card_hight, const_card_width))
        for row in range(0, const_card_hight):     # loop through rows
            for col in range(0, const_card_width): # loop through columns
                """update all cells affected by the current cell and direction"""
                if self.environment.grid[row, col]:    #if cell is NOT occupied by an object/obstacle
                    if self.direction == right:
                        distance = self.environment.distance_grid[row, col, self.direction]
                        prior_update = np.zeros((8))
                        prior_update[3] = prior_update[7] = self.pos_belief[row, col]*0.1
                        prior_update[4] = prior_update[6] = self.pos_belief[row, col] * 0.2
                        prior_update[5] = self.pos_belief[row, col] * 0.4
                        if distance < 7:   #if there is an overshoot
                            prior_update[distance] = sum(prior_update[distance:8])
                            prior_update[distance+1:8] = 0
                        prior_belief_grid[row, col:col+8] += prior_update


        # check if any unoccupied cell is zero:

        return prior_belief_grid


    # self.environment.distance_grid[row, col, self.direction]


    def showPosition(self):
        R = self.environment.grid
        max_belief = np.max(self.pos_belief)
        G = B = (max_belief - self.pos_belief)*self.environment.grid   #for better visualization the maximum occuring belief is always the same red tone RGB = [255,0,0]

        idx_belief_zero = np.argwhere(self.pos_belief == 0)
        num_belief_zero = np.size(idx_belief_zero, 0)
        for i in range(0, num_belief_zero):
            width_value = idx_belief_zero[i,1]
            hight_value = idx_belief_zero[i,0]

            #assign yellow to cells with a belief of ZERO
            G[hight_value,width_value] = 1
            B[hight_value, width_value] = 0


        RGB = np.zeros((const_card_hight,const_card_width,3))
        RGB[:,:,0] = R
        RGB[:,:,1] = G
        RGB[:,:,2] = B

        plt.imshow(RGB) #MxNx3
        plt.show()


def drawEnvironment():
    occupancy_grid = np.ones((const_card_hight,const_card_width),'uint8')

    #draw vertical lines:
    occupancy_grid[:,0] = 0
    occupancy_grid[:,const_card_hight-1] = 0

    occupancy_grid[25:75,const_card_hight/2] = 0

    #draw horizontal lines:
    occupancy_grid[0,:] = 0
    occupancy_grid[const_card_width-1,:] = 0
    occupancy_grid[10, 10:50] = 0
    occupancy_grid[const_card_hight/2, 25:75] = 0
    return occupancy_grid



occupancy_grid = drawEnvironment()
initial_belief_grid = np.ones((const_card_hight,const_card_width))/(sum(sum(occupancy_grid)))
initial_belief_grid = np.multiply(initial_belief_grid,occupancy_grid)

env_map1 = EnvironmentMap(occupancy_grid)
rob1 = Robot(initial_belief_grid, env_map1, [80, 50, 45])
rob1.showPosition()
env_map1.showMap()


