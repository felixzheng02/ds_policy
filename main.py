import numpy as np
import matplotlib.pyplot as plt

from src.util import load_tools, plot_tools
from src.lpvds_class import lpvds_class


input_message = '''
Please choose a data input option:
1. PC-GMM benchmark data
2. LASA benchmark data
3. Damm demo data
4. DEMO
Enter the corresponding option number: '''
# input_opt  = input(input_message)
# input_opt = 4

x, x_dot, x_att, x_init = load_tools.load_data('custom')


# run lpvds
lpvds = lpvds_class(x, x_dot, x_att)
lpvds.begin()


# evaluate results
x_test_list = []
# Do reverse simulation from attractor 4 times

for _ in range(4):
    # First simulate backwards from attractor
    x_start = x_att.copy()
    # Generate direction in positive y and scale to 1cm
    random_dir = np.random.normal(0, 1, x_start.shape[1])  # Random direction in all dimensions
    random_dir[1] = 1.0  # Set y component to 1, others to 0
    random_dir = random_dir / np.linalg.norm(random_dir)  # Normalize to unit vector
    random_dir = random_dir * 0.01  # Scale to 1cm
    x_start += random_dir

    x_reverse = lpvds.sim(x_start, dt=0.01, reverse=True)
    
    # Get the endpoint of reverse simulation as starting point
    x_start_2 = x_reverse[-1:]  # Keep 2D shape by using slice
    
    # Then simulate forwards from that point to attractor 
    x_forward = lpvds.sim(x_start_2, dt=0.01)

    # x_start = x_forward[-1:]
    

    # Add complete trajectory to test list
    x_test_list.append(np.vstack([x_reverse, x_forward]))


# plot results
plot_tools.plot_gmm(x, lpvds.assignment_arr, lpvds.damm)
if x.shape[1] == 2:
    plot_tools.plot_ds_2d(x, x_test_list, lpvds)
else:
    plot_tools.plot_ds_3d(x, x_test_list)
plt.show()