import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# helper functions:
def generate_prey_trajectory(type='linear', start=None, length=3, max_bounds=5):
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
                prey_trajectory = np.linspace(start, end, num=50)
            
            elif type == 'parabolic':
                t = np.linspace(0, 1, num=50)
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
        i = int((delta_theta + fov_angle / 2) / fov_angle * (fov_size - 1))
        j = int((delta_phi + fov_angle / 2) / fov_angle * (fov_size - 1))
        
        # Update the FOV array to indicate the presence of prey
        fov[i, j] += 1
    
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

class Scenario:
    def __init__(self, prey_trajectory, initial_dragonfly_pos, initial_dragonfly_heading):
        """
        Initialize a scenario with prey trajectory and dragonfly parameters.
        
        Parameters:
        - prey_trajectory: numpy array of shape (T, 3) with prey coordinates over time
        - initial_dragonfly_pos: numpy array of shape (3,) with initial dragonfly position
        - initial_dragonfly_heading: numpy array of shape (2,) with heading angles (radians)
        """
        self.prey_trajectory = prey_trajectory
        self.time = 0
        
        # Dragonfly initial state
        self.dragonfly_pos = initial_dragonfly_pos
        self.dragonfly_heading = initial_dragonfly_heading
        
        # Initialize dragonfly trajectory with starting position
        self.dragonfly_trajectory = np.array([initial_dragonfly_pos])
    
    def timestep(self, speed=0.1):
        theta, phi = self.dragonfly_heading
        dx = speed * np.sin(theta) * np.cos(phi)
        dy = speed * np.sin(theta) * np.sin(phi)
        dz = speed * np.cos(theta)
        movement_vector = np.array([dx, dy, dz])
        self.dragonfly_pos += movement_vector
        self.dragonfly_trajectory = np.vstack([self.dragonfly_trajectory, self.dragonfly_pos])
        self.time += 1

        # Check for failstate (out of bounds)
        if np.any(self.dragonfly_pos < 0) or np.any(self.dragonfly_pos > 5):
            print("Failstate: Dragonfly went out of bounds.")
            return False

        # Check for win state (within 0.1 units of prey)
        distance_to_prey = np.linalg.norm(self.dragonfly_pos - self.prey_trajectory[self.time])
        if distance_to_prey < 0.1:
            print("Win state: Dragonfly caught the prey.")
            return True
        
        return None
    
    # controls

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
        
    def plot_scenario_anim(self, brain=None):
        fig = plt.figure(figsize=(15, 8))
        ax3d = fig.add_subplot(121, projection='3d')
        ax2d = fig.add_subplot(122)
        
        def update(frame):
            ax3d.clear()
            ax2d.clear()
            result = self.timestep()
            if result is not None:
                ani.event_source.stop()
                return
            
            # Scatter plot for dragonfly trajectory
            ax3d.scatter(self.dragonfly_trajectory[:, 0], self.dragonfly_trajectory[:, 1], self.dragonfly_trajectory[:, 2], color='lightcoral', s=10, label='Dragonfly Trajectory')
            
            # Scatter plot for prey trajectory
            ax3d.scatter(self.prey_trajectory[:self.time+1, 0], self.prey_trajectory[:self.time+1, 1], self.prey_trajectory[:self.time+1, 2], color='gray', s=10, label='Prey Trajectory')
            
            # Scatter plot for current positions
            ax3d.scatter(self.dragonfly_pos[0], self.dragonfly_pos[1], self.dragonfly_pos[2], color='orangered', s=50, label=f'Dragonfly Position ({self.dragonfly_pos[0]:.2f}, {self.dragonfly_pos[1]:.2f}, {self.dragonfly_pos[2]:.2f})')
            ax3d.scatter(self.prey_trajectory[self.time, 0], self.prey_trajectory[self.time, 1], self.prey_trajectory[self.time, 2], color='black', s=25, label=f'Prey Position ({self.prey_trajectory[self.time, 0]:.2f}, {self.prey_trajectory[self.time, 1]:.2f}, {self.prey_trajectory[self.time, 2]:.2f})')
            
            # Line between dragonfly and prey
            line_x = [self.dragonfly_pos[0], self.prey_trajectory[self.time, 0]]
            line_y = [self.dragonfly_pos[1], self.prey_trajectory[self.time, 1]]
            line_z = [self.dragonfly_pos[2], self.prey_trajectory[self.time, 2]]
            ax3d.plot(line_x, line_y, line_z, color='purple', label=f'Distance: {np.linalg.norm(self.dragonfly_pos - self.prey_trajectory[self.time]):.2f} m')

            theta, phi = self.dragonfly_heading
            dx = np.sin(theta) * np.cos(phi)
            dy = np.sin(theta) * np.sin(phi)
            dz = np.cos(theta)
            ax3d.quiver(self.dragonfly_pos[0], self.dragonfly_pos[1], self.dragonfly_pos[2], dx, dy, dz, length=0.5, color='orangered')
            
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
        
        ani = FuncAnimation(fig, update, frames=range(50), repeat=False)
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
    scenario.plot_scenario_anim()
    
    # Run and plot several time steps
    # for _ in range(100):
    #     state = scenario.timestep()
    #     if state is not None:
    #         break
