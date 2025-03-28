from maddpg import MADDPG
from sim_env import UAVEnv
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_velocity_magnitude(time_steps, velocities_magnitude, output_dir):
    plt.figure(figsize=(15, 4))  
    for i in range(len(velocities_magnitude)):
        if i != 3:
            plt.plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_magnitude[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("Magnitude")
    plt.title("UAV Velocity Magnitude")
    plt.legend()
    plt.grid(True)
    # Save the figure to the output directory
    plt.savefig(os.path.join(output_dir, 'velocity_magnitude.png'))
    plt.close()

def plot_velocity_x(time_steps, velocities_x, output_dir):
    plt.figure(figsize=(15, 4))  
    for i in range(len(velocities_x)):
        if i != 3:
            plt.plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_x[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_x$")
    plt.title("UAV $Vel_x$")
    plt.legend()
    plt.grid(True)
    # Save the figure to the output directory
    plt.savefig(os.path.join(output_dir, 'velocity_x.png'))
    plt.close()

def plot_velocity_y(time_steps, velocities_y, output_dir):
    plt.figure(figsize=(15, 4))  
    for i in range(len(velocities_y)):
        if i != 3:
            plt.plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_y[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_y$")
    plt.title("UAV $Vel_y$")
    plt.legend()
    plt.grid(True)
    # Save the figure to the output directory
    plt.savefig(os.path.join(output_dir, 'velocity_y.png'))
    plt.close()

def plot_velocities(velocities_magnitude, velocities_x, velocities_y, output_dir):
    time_steps = range(len(velocities_magnitude[0]))
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    for i in range(len(velocities_magnitude)):
        if i != 3:
            axs[0].plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            axs[0].plot(time_steps, velocities_magnitude[i], label='Target')
    axs[0].set_title('Speed Magnitude vs Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Speed Magnitude')
    axs[0].legend()

    for i in range(len(velocities_x)):
        if i != 3:
            axs[1].plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            axs[1].plot(time_steps, velocities_x[i], label='Target')
    axs[1].set_title('Velocity X Component vs Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Velocity X Component')
    axs[1].legend()

    for i in range(len(velocities_y)):
        if i != 3:
            axs[2].plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            axs[2].plot(time_steps, velocities_y[i], label='Target')
    axs[2].set_title('Velocity Y Component vs Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Velocity Y Component')
    axs[2].legend()

    plt.tight_layout()
    # Save the combined figure to the output directory
    plt.savefig(os.path.join(output_dir, 'combined_velocities.png'))
    plt.close()


if __name__ == '__main__':
    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 2
    actor_dims = []
    velocities_magnitude = [[] for _ in range(env.num_agents)]  # record magnitude of vel
    velocities_x = [[] for _ in range(env.num_agents)]  # record vel_x
    velocities_y = [[] for _ in range(env.num_agents)]  # record vel_y

    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=128, fc2=128, alpha=0.0001, beta=0.003, scenario='UAV_Round_up',
                           chkpt_dir='tmp/maddpg/')
    
    maddpg_agents.load_checkpoint()
    print('---- Evaluating ----')

    obs = env.reset()
    output_dir = 'frames_output'
    output_speed_dir = 'speed_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_speed_dir):
        os.makedirs(output_speed_dir)

    total_steps = 0

    for frame in range(10000):  # Loop for each frame
        for i in range(env.num_agents):
            vel = env.multi_current_vel[i]
            v_x, v_y = vel
            speed = np.linalg.norm(vel)

            velocities_magnitude[i].append(speed)
            velocities_x[i].append(v_x)
            velocities_y[i].append(v_y)

        actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
        obs_, _, dones = env.step(actions)
        env.render_anime(frame)
        obs = obs_

        # Save the frame as an image
        frame_filename = os.path.join(output_dir, f'frame_{frame}.png')
        plt.savefig(frame_filename)  # Save current frame as an image

      
        
        if any(dones):
            print("Round-up finished in", frame, "steps.")
            smoothed_velocities_magnitude = [[] for _ in range(env.num_agents)]
            smoothed_velocities_x = [[] for _ in range(env.num_agents)]  
            smoothed_velocities_y = [[] for _ in range(env.num_agents)] 
            for i in range(env.num_agents):
                _velocity_magnitude = moving_average(velocities_magnitude[i], window_size=5)
                _velocity_x = moving_average(velocities_x[i], window_size=5)
                _velocity_y = moving_average(velocities_y[i], window_size=5)
                smoothed_velocities_magnitude[i] = _velocity_magnitude
                smoothed_velocities_x[i] = _velocity_x
                smoothed_velocities_y[i] = _velocity_y
            plot_velocities(smoothed_velocities_magnitude, smoothed_velocities_x, smoothed_velocities_y,output_speed_dir)
            time_steps = range(len(smoothed_velocities_magnitude[0]))
            plot_velocity_magnitude(time_steps, smoothed_velocities_magnitude,output_speed_dir)
            plot_velocity_x(time_steps, smoothed_velocities_x,output_speed_dir)
            plot_velocity_y(time_steps, smoothed_velocities_y,output_speed_dir)
            break  # End the loop once the episode is finished
