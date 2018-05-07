import numpy as np
import matplotlib.pyplot as plt
right, up, left, down = 0, 1, 2, 3

class EnvironmentMap():
    def __init__(self, occupancy_grid):
        self.grid = occupancy_grid
        self.card_height = np.size(occupancy_grid, axis=0)
        self.card_width = np.size(occupancy_grid, axis=1)
        self.num_cells =self.card_height*self.card_width
        self.dist_grid = self._createDistanceGrid()
        self.num_occupied_cells = self.num_cells - np.count_nonzero(self.grid)
        self.epsilon = 1.0/(self.num_cells*100)    #minimum belief value one cell can have

    def _createDistanceGrid(self):
        distance_grid = np.zeros((self.card_height, self.card_width, 4))
        """update the distances in the direction "right" and "left" of the cells in grid[row,col_prev:col"""
        for row in range(0, self.card_height):     # loop through rows
            col_found = False
            for col in range(0, self.card_width): # loop through columns
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
                        col_found = True  # at least one occupied cell was found already


        """update the distances in the vertical directions "up" and "down" of the cells in grid[prev_row:row,col]"""
        for col in range(0, self.card_width):     # loop through columns
            row_found = False   # occupied row in current column not found in the beginning
            for row in range(0, self.card_height): # loop through rows
                if self.grid[row, col] == 0:  #if the cell IS occupied --> calculate the distances for each vertical direction

                    # if another occupied cell is found in the current column
                    # --> calculate the up and/or down distances for each cell
                    if row_found == True:
                        for update_row in range(prev_row+1, row):
                            distance_grid[update_row, col, down]  = row - update_row -1       #calculate distance to right object
                            distance_grid[update_row, col, up]    = update_row - prev_row - 1  #calculate distance to left object
                        prev_row = row  # set current bottom object to prev_object and continue looping from top to bottom
                    else:
                        prev_row = row  #first occupied cell found (in the current column)
                        row_found = True  # at least one occupied cell found

        return distance_grid

    def showMap(self):
        plt.imshow(self.grid)
        plt.show()


def drawEnvironment(height, width):
    occupancy_grid = np.ones((height,width),'uint8')

    #draw vertical lines:
    occupancy_grid[:,0] = 0
    occupancy_grid[:,height-1] = 0

    occupancy_grid[25:75, height/2] = 0

    #draw horizontal lines:
    occupancy_grid[0,:] = 0
    occupancy_grid[width-1,:] = 0
    occupancy_grid[10, 10:50] = 0
    occupancy_grid[height/2, 25:75] = 0
    return occupancy_grid


class RealPosition:
    def __init__(self, x_position, y_position, environment_map):
        self.x = x_position
        self.y = y_position
        self.environment_map = environment_map

    def kidnapp(self, x_position, y_position):
        if self.environment_map.grid[y_position,x_position] == 1: #check if cell is free
            self.x = x_position
            self.y = y_position
        else:
            print "This cell is occupied already!\n Robot couldn't be kidnapped"

    def updatePosition(self, delta_x, delta_y):
        self.x += delta_x
        self.y += delta_y
        #print "x = %d  y = %d" % (self.x, self.y)

    def getPosition(self):
        return int(self.x), int(self.y)


class Robot:
    """ This is the class of the robot:
        The roboter has a position and a direction
    """

    def __init__(self, initial_belief_grid, environment_map, real_x_pos, real_y_pos):
        self.pos_belief = initial_belief_grid
        self.environment = environment_map
        self.direction = right
        self.real_position = RealPosition(real_x_pos, real_y_pos, environment_map)

    def move(self, number_of_steps, direction):
        plt.figure(1)
        if self.direction == right:
            direction_string = 'Right'
        elif self.direction == left:
            direction_string = 'Left'
        elif self.direction == up:
            direction_string = 'Top'
        elif self.direction == down:
            direction_string = 'Bottom'
        plt.title('Belief Grid Before the Next %d Steps to the %s' % (number_of_steps, direction_string))
        self.showPosition()
        for i in range(0,number_of_steps):
            self.direction = direction
            moving_distance = np.random.choice([3, 4, 5, 6, 7], 1, p=[0.1, 0.2, 0.4, 0.2, 0.1])
            x,y = self.real_position.getPosition()
            distance_to_obstacle = self.environment.dist_grid[y, x, self.direction]
            if distance_to_obstacle < moving_distance:
                moving_distance = distance_to_obstacle
                
            if self.direction == right:
                self.real_position.updatePosition(moving_distance, 0)
            elif self.direction == left:
                self.real_position.updatePosition(-moving_distance, 0)
            elif self.direction == up:
                self.real_position.updatePosition(0, -moving_distance)
            elif self.direction == down:
                self.real_position.updatePosition(0, moving_distance)

            self.pos_belief = self._calcPriorBelief()   # calculate prior belief
            plt.figure(2)
            plt.title('Prior Beliefe of Step %d out of %d Steps to the %s' % (i+1, number_of_steps, direction_string))
            self.showPosition()                         # show prior belief grid
            self.pos_belief = self._calcPosteriorBelief()
        plt.figure(3)
        plt.title('Belief Grid After the Next %d Steps to the %s' % (number_of_steps, direction_string))
        self.showPosition()

    # in reality this method would provide the measured distance values,
    # in order to emulate the measurement aberration the measurement model is modeled
    def _getDistance(self):
        x_real, y_real = self.real_position.getPosition()
        measurement_model = [-2, -1, 0, 1, 2] + self.environment.dist_grid[y_real, x_real, self.direction]
        measured_distance = np.random.choice(measurement_model, size=1, p=[0.1, 0.2, 0.4, 0.2, 0.1])
        if measured_distance < 0:
            measured_distance = 0
        return measured_distance

    def _calcPriorBelief(self):
        """In this function the current belief grid is supposed to have at least a
        minimum belief "epsilon" in each unoccupied cell in order to detect a potential kidnapping
        Before the prior_belief_grid gets returned it is taken care of the minimum probability
        of a single cell to be more than "epsilon" """
        prior_belief_grid = np.zeros((self.environment.card_height, self.environment.card_width))
        for row in range(0, self.environment.card_height):  # loop through rows
            for col in range(0, self.environment.card_width):  # loop through columns
                """update all cells affected by the current cell and direction"""
                if self.environment.grid[row, col]: # if cell is NOT occupied by an object/obstacle:
                    prior_update = np.zeros((8))    # prior_update holds the probabilities to be added
                    prior_update[3] = prior_update[7] = self.pos_belief[row, col] * 0.1
                    prior_update[4] = prior_update[6] = self.pos_belief[row, col] * 0.2
                    prior_update[5] = self.pos_belief[row, col] * 0.4

                    distance = int(self.environment.dist_grid[row, col, self.direction])
                    max_drive_distance = 7
                    if distance < 7:  # if there is an overshoot (obstacle within 7 cells)
                        prior_update[distance] = sum(prior_update[distance:8])
                        prior_update = prior_update[0:distance+1]
                        max_drive_distance = distance # +1 to consider the current position too

                    if self.direction == right:
                        prior_belief_grid[row, col:(col+max_drive_distance+1)] += prior_update
                    elif self.direction == left:
                        prior_belief_grid[row, (col-max_drive_distance):col+1] += np.flipud(prior_update)
                    elif self.direction == up:
                        prior_belief_grid[(row-max_drive_distance):row+1, col] += np.flipud(prior_update)
                    elif self.direction == down:
                        prior_belief_grid[row:(row+max_drive_distance+1), col] += prior_update

        # check if any unoccupied cell is less than "epsilon":
        idx_belief_too_low = np.argwhere(prior_belief_grid+abs(self.environment.grid-1) < self.environment.epsilon)

        for i in range(0, np.size(idx_belief_too_low, 0)):
            prior_belief_grid[idx_belief_too_low[i,0],idx_belief_too_low[i,1]] = self.environment.epsilon*1.01 # chosen slightly higher because normalization follows

        prior_belief_grid /= sum(sum(prior_belief_grid))        # normalize to total probability of 1
        return prior_belief_grid


    def _calcPosteriorBelief(self):
        measured_distance = self._getDistance()
        distance_belief_grid = np.zeros((self.environment.card_height, self.environment.card_width))
        # for every cell calculate the possibility according to the measured distance:
        for row in range(0, self.environment.card_height):  # loop through rows
            for col in range(0, self.environment.card_width):  # loop through columns
                aberration = abs(measured_distance - self.environment.dist_grid[row, col, self.direction])
                # If the distance in the current direction is in the range of +-2 cells:
                if aberration <= 2:
                    if aberration == 2:
                        distance_belief_grid[row, col] = 0.1
                    elif aberration == 1:
                        distance_belief_grid[row, col] = 0.2
                    elif aberration == 0:
                        distance_belief_grid[row, col] = 0.4
        distance_belief_grid /= sum(sum(distance_belief_grid))  #normalize to total probability = 1

        # ensure that the cells have at least the value "epsilon":
        idx_belief_too_low = np.argwhere(distance_belief_grid+abs(self.environment.grid-1) < self.environment.epsilon)
        for i in range(0, np.size(idx_belief_too_low, 0)):
            distance_belief_grid[idx_belief_too_low[i,0],idx_belief_too_low[i,1]] = self.environment.epsilon*1.01 # chosen slightly higher because normalization follows

        distance_belief_grid /= sum(sum(distance_belief_grid))        # normalize to total probability of 1
        posterior_belief = (distance_belief_grid * self.pos_belief)
        posterior_belief /= sum(sum(posterior_belief))
        return posterior_belief

    def showPosition(self):
        self._showPosition(self.pos_belief)

    def _showPosition(self, belief_grid):
        R = self.environment.grid
        max_belief = np.max(belief_grid)
        min_belief = np.min(belief_grid)
        GB_transformation = (max_belief - belief_grid)/(max_belief-min_belief) * self.environment.grid #Normalize green and blue value between 0 and 1
        G = GB_transformation  # for better visualization the maximum occuring belief is always the same red tone RGB = [255,0,0]
        B = GB_transformation  # for better visualization the maximum occuring belief is always the same red tone RGB = [255,0,0]

        idx_belief_zero = np.argwhere(belief_grid+abs(self.environment.grid-1) < self.environment.epsilon)
        num_belief_zero = np.size(idx_belief_zero, 0)
        for i in range(0, num_belief_zero):
            # assign yellow to cells with a belief of ZERO
            B[idx_belief_zero[i, 0], idx_belief_zero[i, 1]] = 0.0
            G[idx_belief_zero[i, 0], idx_belief_zero[i, 1]] = 1.0

        #G = G*self.environment.grid     # set the occupied cells back to zero
        RGB = np.zeros((self.environment.card_height, self.environment.card_width, 3))
        RGB[:, :, 0] = R
        RGB[:, :, 1] = G
        RGB[:, :, 2] = B
        x,y = self.real_position.getPosition()
        RGB[y, x,:] = [0,0,0.5] #mark robots real position
        plt.imshow(RGB)  # MxNx3
        plt.show()

def main():
    card_height = 100
    card_width  = 100
    occupancy_grid = drawEnvironment(card_height,card_width)
    initial_belief_grid = np.ones((card_height,card_width))/(card_height*card_width)
    initial_belief_grid = np.multiply(initial_belief_grid,occupancy_grid)

    env_map1 = EnvironmentMap(occupancy_grid)
    plt.title('Map Depicting the Robots World')
    plt.text(5, 95, 'The following graphs show the robot\'s location probability\n'
                    'The belief is depicted in red hue. \n'
                    'A cell with a higher hue corresponds to \n'
                    'a higher probability of the robot to be in the cell\n'
                    'The robots real position is depicted with a blue pixel.\n'
                    'The black pixels visualize occupied cells.', fontsize=9)
    env_map1.showMap()
    rob1 = Robot(initial_belief_grid, env_map1, 50, 3)
    rob1.move(5, up)
    rob1.move(1, left)
    rob1.move(2, down)
    rob1.real_position.kidnapp(40,3)
    rob1.move(4,right)
    rob1.move(3,up)
    rob1.move(1,left)
    rob1.move(2, down)

main()

