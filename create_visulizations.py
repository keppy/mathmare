import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def create_limit_visualization(save_path='limit_approach.png'):
    """Create a clean limit visualization"""
    # Set style for clean, modern look
    plt.style.use('seaborn-v0_8-white')  # Using 'white' instead of 'seaborn-white'

    # Create figure with high DPI for quality
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # Set background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Create data
    x = np.linspace(-2, 6, 1000)
    def f(x): return (x-2)**2 / 2 + 1
    y = f(x)

    # Plot main function
    ax.plot(x, y, color='#4A90E2', linewidth=2.5, zorder=1)  # Using web-safe blue

    # Add approach arrows
    ax.arrow(1, f(1), 0.5, f(2)-f(1), head_width=0.1,
            head_length=0.1, fc='#2ECC71', ec='#2ECC71',  # Using web-safe green
            length_includes_head=True)
    ax.arrow(3, f(3), -0.5, f(2)-f(3), head_width=0.1,
            head_length=0.1, fc='#2ECC71', ec='#2ECC71',
            length_includes_head=True)

    # Add limit point
    ax.plot(2, f(2), 'o', color='#E74C3C', markersize=10, zorder=2)  # Using web-safe red

    # Clean up plot
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Set limits and aspect ratio
    ax.set_xlim(-3, 7)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

def create_derivative_visualization(save_path='derivative_intro.png'):
    """Create a visualization of derivative as tangent line"""
    plt.style.use('seaborn-v0_8-white')

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Create data for parabola
    x = np.linspace(-2, 4, 1000)
    def f(x): return x**2
    y = f(x)

    # Point of tangency
    x0, y0 = 1, f(1)

    # Plot main function
    ax.plot(x, y, color='#4A90E2', linewidth=2.5, zorder=1)

    # Add tangent line
    derivative_at_x0 = 2*x0  # Derivative of x^2 is 2x
    def tangent(x):
        return derivative_at_x0 * (x - x0) + y0

    # Plot tangent line
    x_tangent = np.linspace(0, 2, 100)
    ax.plot(x_tangent, tangent(x_tangent), color='#E74C3C',
            linewidth=2, linestyle='-', zorder=2)

    # Add point of tangency
    ax.plot(x0, y0, 'o', color='#2ECC71', markersize=10, zorder=3)

    # Clean up plot
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Set limits and aspect ratio
    ax.set_xlim(-2, 4)
    ax.set_ylim(-1, 6)
    ax.set_aspect('equal')

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

# Test both functions
create_limit_visualization()
create_derivative_visualization()
