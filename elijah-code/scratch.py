import numpy as np
import random as rd
import os
import matplotlib.pyplot as plt
from datetime import datetime as dt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Constants
NOISE = 0.05



# helper functions:
def generate_prey_trajectory(type='linear', start=None, length=4, max_bounds=5, numsize = 50):
    """
    Generate a prey trajectory.
    
    Parameters:
    - type: 'linear' or 'parabolic'
    - start: starting coordinates (optional, default is random between 1 and 4 in each dimension)
    - length: minimum length of the trajectory
    - max_bounds: maximum bounds for the trajectory (default is 5)
    
    Returns:
    - prey_trajectory: numpy array of shape (T, 3) with prey coordinates over time
    """
    cont = True
    while cont:
        try:
            if start is None:
                start = np.random.uniform(1, 4, size=3)
            
            if type == 'linear':
                direction = np.random.uniform(-1, 1, size=3)
                direction /= np.linalg.norm(direction)  # Normalize direction
                end = start + direction * length
                end = np.clip(end, 0, max_bounds)
                prey_trajectory = np.linspace(start, end, numsize)
            
            elif type == 'parabolic':
                t = np.linspace(0, 1, numsize)
                direction = np.random.uniform(-1, 1, size=3)
                direction /= np.linalg.norm(direction)  # Normalize direction
                end = start + direction * length
                end = np.clip(end, 0, max_bounds)
                a = (end - start) / (t[-1]**2)
                prey_trajectory = start + a * t[:, np.newaxis]**2
            
            # Ensure the trajectory is within bounds and at least 3 meters in length
            distances = np.linalg.norm(np.diff(prey_trajectory, axis=0), axis=1)
            total_distance = np.cumsum(distances)
            if total_distance[-1] < length:
                raise ValueError("Generated trajectory is less than the required length.")
            
            prey_trajectory = prey_trajectory[np.insert(total_distance <= length, 0, True)]
            cont = False
        except ValueError:
            cont = True
    
    return prey_trajectory

def generate_dragonfly_initial_state(prey_position):
    """
    Generate a random initial state for the dragonfly.
    
    Parameters:
    - prey_position: numpy array of shape (3,) with prey's current position
    
    Returns:
    - initial_pos: numpy array of shape (3,) with initial dragonfly position
    - initial_heading: numpy array of shape (2,) with heading angles (theta, phi) in radians
    """

    initial_pos = np.random.uniform(0, 5, size=3)
    initial_heading = np.array([np.pi/4, np.pi/4])  # 45-degree angles
    return initial_pos, initial_heading


def calculate_fov(dragonfly_heading, dragonfly_pos, prey_pos, fov_size=21, fov_angle=np.pi):
    """
    Calculate the dragonfly's field of view (FOV).
    
    Parameters:
    - dragonfly_heading: numpy array of shape (2,) with heading angles (theta, phi) in radians
    - dragonfly_pos: numpy array of shape (3,) with dragonfly's current position
    - prey_pos: numpy array of shape (3,) with prey's current position
    - fov_size: size of the FOV array (default is 21x21)
    - fov_angle: field of view angle in radians (default is 180 degrees)
    
    Returns:
    - fov: 2D numpy array of shape (fov_size, fov_size) representing the FOV
    """
    fov = np.zeros((fov_size, fov_size))
    
    # Calculate relative position of prey
    relative_pos = prey_pos - dragonfly_pos
    
    # Convert relative position to spherical coordinates
    r = np.linalg.norm(relative_pos)
    theta = np.arccos(relative_pos[2] / r) if r != 0 else 0
    phi = np.arctan2(relative_pos[1], relative_pos[0])
    
    # Convert dragonfly heading to spherical coordinates
    heading_theta, heading_phi = dragonfly_heading
    
    # Calculate the angle difference between the heading and the prey position
    delta_theta = theta - heading_theta
    delta_phi = phi - heading_phi
    
    # Normalize the angles to be within the FOV
    if np.abs(delta_theta) <= fov_angle / 2 and np.abs(delta_phi) <= fov_angle / 2:
        # Map the angles to the FOV array indices
        i_center = int((delta_theta + fov_angle / 2) / fov_angle * (fov_size - 1))
        j_center = int((delta_phi + fov_angle / 2) / fov_angle * (fov_size - 1))
        
        # Determine intensity based on distance
        intensity = max(0.5, min(1, 1 - (r / 5)))
        
        # Spread the intensity across multiple indices
        spread = max(1, int((1 - r / 5) * (fov_size // 4)))
        # print(spread)
        for di in range(-spread, spread + 1):
            for dj in range(-spread, spread + 1):
                i = i_center + di 
                j = j_center + dj
                if 0 <= i < fov_size and 0 <= j < fov_size:
                    distance_factor = max(0, 1 - (np.sqrt(di**2 + dj**2) / spread))
                    fov[i, j] += intensity * distance_factor
    
    # add noise (only up to ) to fov: 
    fov += np.random.normal(0, NOISE, fov.shape)
    
    return fov

# brains:

def brain_classic_direct(fov):
    """
    Determine the pitch and yaw adjustments to center the largest index in the FOV.
    
    Parameters:
    - fov: 2D numpy array representing the field of view
    
    Returns:
    - (pitch, yaw): tuple where pitch and yaw are either 1, 0, or -1
    """
    fov_size = fov.shape[0]
    center = fov_size // 2
    
    # Find the indices of the maximum value in the FOV
    max_index = np.unravel_index(np.argmax(fov), fov.shape)
    max_i, max_j = max_index
    
    # Determine pitch adjustment
    if max_i < center:
        pitch = -1
    elif max_i > center:
        pitch = 1
    else:
        pitch = 0
    
    # Determine yaw adjustment
    if max_j < center:
        yaw = -1
    elif max_j > center:
        yaw = 1
    else:
        yaw = 0
    
    return (pitch, yaw)

def brain_offset_prev(fov, prev_fov=None):
    """
    Determine the pitch and yaw adjustments to predict prey movement.
    
    Parameters:
    - fov: 2D numpy array representing the current field of view
    - prev_fov: 2D numpy array representing the previous field of view (optional)
    
    Returns:
    - (pitch, yaw): tuple where pitch and yaw are either 1, 0, or -1
    """
    fov_size = fov.shape[0]
    center = fov_size // 2
    
    # Find the indices of the maximum value in the current FOV
    max_index = np.unravel_index(np.argmax(fov), fov.shape)
    max_i, max_j = max_index
    
    # If previous FOV is provided, calculate movement prediction
    if prev_fov is not None:
        prev_max_index = np.unravel_index(np.argmax(prev_fov), prev_fov.shape)
        prev_i, prev_j = prev_max_index
        
        # Predict movement direction
        i_delta = max_i - prev_i
        j_delta = max_j - prev_j
        
        # Predict next position with some anticipation
        predicted_i = max_i + i_delta
        predicted_j = max_j + j_delta
        
        # Determine pitch adjustment based on predicted position
        if predicted_i < center - fov_size * 0.1:
            pitch = -1
        elif predicted_i > center + fov_size * 0.1:
            pitch = 1
        else:
            pitch = 0
        
        # Determine yaw adjustment based on predicted position
        if predicted_j < center - fov_size * 0.1:
            yaw = -1
        elif predicted_j > center + fov_size * 0.1:
            yaw = 1
        else:
            yaw = 0
    
    else:
        # If no previous FOV, use current FOV position
        if max_i < center - fov_size * 0.1:
            pitch = -1
        elif max_i > center + fov_size * 0.1:
            pitch = 1
        else:
            pitch = 0
        
        if max_j < center - fov_size * 0.1:
            yaw = -1
        elif max_j > center + fov_size * 0.1:
            yaw = 1
        else:
            yaw = 0
    
    return (pitch, yaw)

def brain_offset(fov):
    """
    Determine the pitch and yaw adjustments to center the largest index in the FOV.
    
    Parameters:
    - fov: 2D numpy array representing the field of view
    
    Returns:
    - (pitch, yaw): tuple where pitch and yaw are either 1, 0, or -1
    """
    fov_size = fov.shape[0]
    center = fov_size // 2
    
    # Find the indices of the maximum value in the FOV
    max_index = np.unravel_index(np.argmax(fov), fov.shape)
    max_i, max_j = max_index
    
    # Calculate offset from center
    pitch_offset = max_i - center
    yaw_offset = max_j - center
    
    # Determine pitch adjustment with a threshold
    if pitch_offset < -fov_size * 0.1:  # 10% threshold
        pitch = -1
    elif pitch_offset > fov_size * 0.1:
        pitch = 1
    else:
        pitch = 0
    
    # Determine yaw adjustment with a threshold
    if yaw_offset < -fov_size * 0.1:
        yaw = -1
    elif yaw_offset > fov_size * 0.1:
        yaw = 1
    else:
        yaw = 0
    
    return (pitch, yaw)

class Scenario:
    def __init__(self, prey_trajectory, initial_dragonfly_pos, initial_dragonfly_heading, brain=brain_classic_direct):
        """
        Initialize a scenario with prey trajectory and dragonfly parameters.
        
        Parameters:
        - prey_trajectory: numpy array of shape (T, 3) with prey coordinates over time
        - initial_dragonfly_pos: numpy array of shape (3,) with initial dragonfly position
        - initial_dragonfly_heading: numpy array of shape (2,) with heading angles (radians)
        """
        self.prey_trajectory = prey_trajectory
        self.time = 0

        self.brain = brain
        
        # Dragonfly initial state
        self.dragonfly_pos = initial_dragonfly_pos
        self.dragonfly_heading = initial_dragonfly_heading
        
        # Initialize dragonfly trajectory with starting position
        self.dragonfly_trajectory = np.array([initial_dragonfly_pos])
    
    def timestep(self, speed=0.2):
        if self.time >= len(self.prey_trajectory):
            return "end"
        
        theta, phi = self.dragonfly_heading
        dx = speed * np.sin(theta) * np.cos(phi)
        dy = speed * np.sin(theta) * np.sin(phi)
        dz = speed * np.cos(theta)
        movement_vector = np.array([dx, dy, dz])
        self.dragonfly_pos += movement_vector
        self.dragonfly_trajectory = np.vstack([self.dragonfly_trajectory, self.dragonfly_pos])
        self.time += 1

        # turn
        fov = calculate_fov(self.dragonfly_heading, self.dragonfly_pos, self.prey_trajectory[self.time])
        past_fov = calculate_fov(self.dragonfly_heading, self.dragonfly_pos, self.prey_trajectory[self.time-1])
        fov = fov + (past_fov * 0.2)  # add a little bit of memory
        if self.brain == brain_offset_prev:
            self.change_heading(self.brain(fov, past_fov))
        else:
            self.change_heading(self.brain(fov))

        # prey finished
        if self.time >= len(self.prey_trajectory):
            print("Failstate: Prey escaped.")
            return False

        # Check for failstate (out of bounds)
        if np.any(self.dragonfly_pos < 0) or np.any(self.dragonfly_pos > 5):
            print("Failstate: Dragonfly went out of bounds.")
            return False

        # Check for win state (within 0.1 units of prey)
        distance_to_prey = np.linalg.norm(self.dragonfly_pos - self.prey_trajectory[self.time])
        if distance_to_prey < 0.2:
            print("Win state: Dragonfly caught the prey.")
            return True
        
        return None

    # controls
    def change_heading(self, pitch_yaw_tuple):
        pitch, yaw = pitch_yaw_tuple
        if pitch == 1:
            self.pitch_up()
        elif pitch == -1:
            self.pitch_down()
        if yaw == 1:
            self.yaw_right()
        elif yaw == -1:
            self.yaw_left()

    def pitch_up(self, angle=np.pi/12):
        theta, phi = self.dragonfly_heading
        theta = np.clip(theta + angle, 0, np.pi)
        self.dragonfly_heading = np.array([theta, phi])
    
    def pitch_down(self, angle=np.pi/12):
        theta, phi = self.dragonfly_heading
        theta = np.clip(theta - angle, 0, np.pi)
        self.dragonfly_heading = np.array([theta, phi])

    def yaw_left(self, angle=np.pi/12):
        theta, phi = self.dragonfly_heading
        phi = (phi - angle) % (2 * np.pi)
        self.dragonfly_heading = np.array([theta, phi])

    def yaw_right(self, angle=np.pi/12):
        theta, phi = self.dragonfly_heading
        phi = (phi + angle) % (2 * np.pi)
        self.dragonfly_heading = np.array([theta, phi])
        
    def plot_scenario(self, save=False):
        fig = plt.figure(figsize=(15, 8))
        ax3d = fig.add_subplot(121, projection='3d')
        ax2d = fig.add_subplot(122)
        
        def update(frame):
            result = self.timestep()
            if result == "end" or result is not None:
                ani.event_source.stop()
                plot_final_state()
                return
            
            plot_current_state()
        
        def plot_current_state():
            ax3d.clear()
            ax2d.clear()
            
            # Scatter plot for dragonfly trajectory
            ax3d.scatter(self.dragonfly_trajectory[:, 0], self.dragonfly_trajectory[:, 1], self.dragonfly_trajectory[:, 2], color='thistle', s=10, label='Dragonfly Trajectory')
            
            # Scatter plot for prey trajectory
            ax3d.scatter(self.prey_trajectory[:self.time+1, 0], self.prey_trajectory[:self.time+1, 1], self.prey_trajectory[:self.time+1, 2], color='lightgreen', s=10, label='Prey Trajectory')
            
            # Scatter plot for current positions
            ax3d.scatter(self.dragonfly_pos[0], self.dragonfly_pos[1], self.dragonfly_pos[2], color='darkorchid', s=50, label=f'Dragonfly Position ({self.dragonfly_pos[0]:.2f}, {self.dragonfly_pos[1]:.2f}, {self.dragonfly_pos[2]:.2f})')
            ax3d.scatter(self.prey_trajectory[self.time, 0], self.prey_trajectory[self.time, 1], self.prey_trajectory[self.time, 2], color='forestgreen', s=25, label=f'Prey Position ({self.prey_trajectory[self.time, 0]:.2f}, {self.prey_trajectory[self.time, 1]:.2f}, {self.prey_trajectory[self.time, 2]:.2f})')
            
            # Line between dragonfly and prey
            line_x = [self.dragonfly_pos[0], self.prey_trajectory[self.time, 0]]
            line_y = [self.dragonfly_pos[1], self.prey_trajectory[self.time, 1]]
            line_z = [self.dragonfly_pos[2], self.prey_trajectory[self.time, 2]]
            ax3d.plot(line_x, line_y, line_z, color='lightcoral', label=f'Distance: {np.linalg.norm(self.dragonfly_pos - self.prey_trajectory[self.time]):.2f} m')

            theta, phi = self.dragonfly_heading
            dx = np.sin(theta) * np.cos(phi)
            dy = np.sin(theta) * np.sin(phi)
            dz = np.cos(theta)
            ax3d.quiver(self.dragonfly_pos[0], self.dragonfly_pos[1], self.dragonfly_pos[2], dx, dy, dz, length=0.5, color='darkorchid')
            
            ax3d.set_xlim(0, 5)
            ax3d.set_ylim(0, 5)
            ax3d.set_zlim(0, 5)
            ax3d.set_xlabel('X (m)')
            ax3d.set_ylabel('Y (m)')
            ax3d.set_zlabel('Z (m)')
            ax3d.set_title(f'Scenario at Time Step {self.time}')
            ax3d.legend()
            
            # Calculate and plot FOV heatmap
            fov = calculate_fov(self.dragonfly_heading, self.dragonfly_pos, self.prey_trajectory[self.time])
            ax2d.imshow(fov, cmap='Blues', interpolation='nearest')
            ax2d.set_title('Dragonfly FOV')
            ax2d.set_xlabel('Phi')
            ax2d.set_ylabel('Theta')
        
        def plot_final_state():
            plot_current_state()
            plt.draw()
        
        ani = FuncAnimation(fig, update, frames=range(50), repeat=False)

        if save:
            output_dir = "scenario_out_3d"
            os.makedirs(output_dir, exist_ok=True)
            current_time = dt.now().strftime("%Y%m%d_%H%M%S")
            gif_path = os.path.join(output_dir, f"animation_{current_time}.gif")
            ani.save(gif_path, writer=PillowWriter(fps=10))

        plt.show() 

# Example usage:
if __name__ == "__main__":
    # Generate example trajectories
    # np.random.seed(42)
    
    # Prey trajectory: simple random walk
    # prey_traj = np.linspace([1.5, 1.5, 1.5], [3.5, 3.5, 3.5], num=100)
    # prey_traj = np.clip(prey_traj, 0, 5)  # Constrain to 5x5x5 space
    prey_traj = generate_prey_trajectory(type="parabolic")
    
    # Initial dragonfly position and heading
    initial_pos = np.array([0, 0, 0], dtype=np.float64)
    initial_heading = np.array([np.pi/4, np.pi/4])  # 45-degree angles
    
    # Create scenario
    scenario = Scenario(prey_traj, initial_pos, initial_heading)
    scenario.plot_scenario(save=False)
        
    prey_traj = generate_prey_trajectory(type="parabolic")
    
    # Initial dragonfly position and heading
    initial_pos = np.array([0, 0, 0], dtype=np.float64)
    initial_heading = np.array([np.pi/4, np.pi/4])  # 45-degree angles

    scenario = Scenario(prey_traj, initial_pos, initial_heading, brain=brain_offset_prev)
    scenario.plot_scenario(save=False)

    prey_traj = generate_prey_trajectory(type="parabolic")
    
    # Initial dragonfly position and heading
    initial_pos = np.array([0, 0, 0], dtype=np.float64)
    initial_heading = np.array([np.pi/4, np.pi/4])  # 45-degree angles

    scenario = Scenario(prey_traj, initial_pos, initial_heading, brain=brain_offset)
    scenario.plot_scenario(save=False)    
    # Run and plot several time steps
    # for _ in range(100):
    #     state = scenario.timestep()
    #     if state is not None:
    #         break
