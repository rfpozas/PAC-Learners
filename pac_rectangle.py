"""
PAC Learning Rectangle Animation
Generates an animated GIF showing how PAC learning converges to the true concept.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

# Set random seed for reproducibility
np.random.seed(42)

# Define the true (unknown) rectangle
true_rect = {
    'x_min': 2.0,
    'x_max': 8.0,
    'y_min': 3.0,
    'y_max': 7.0
}

def is_inside(x, y, rect):
    """Check if a point is inside the rectangle."""
    return (rect['x_min'] <= x <= rect['x_max'] and 
            rect['y_min'] <= y <= rect['y_max'])

def compute_learned_rect(x_pos, y_pos):
    """Compute the tightest rectangle containing all positive examples."""
    if len(x_pos) == 0:
        return None
    return {
        'x_min': min(x_pos),
        'x_max': max(x_pos),
        'y_min': min(y_pos),
        'y_max': max(y_pos)
    }

def compute_error(learned_rect, true_rect):
    """Compute normalized error between learned and true rectangle."""
    if learned_rect is None:
        return 1.0
    
    true_area = (true_rect['x_max'] - true_rect['x_min']) * (true_rect['y_max'] - true_rect['y_min'])
    learned_area = (learned_rect['x_max'] - learned_rect['x_min']) * (learned_rect['y_max'] - learned_rect['y_min'])
    
    # Intersection
    x_overlap = max(0, min(true_rect['x_max'], learned_rect['x_max']) - 
                       max(true_rect['x_min'], learned_rect['x_min']))
    y_overlap = max(0, min(true_rect['y_max'], learned_rect['y_max']) - 
                       max(true_rect['y_min'], learned_rect['y_min']))
    intersection = x_overlap * y_overlap
    
    # Symmetric difference normalized by true area
    error = (true_area + learned_area - 2 * intersection) / true_area
    return error

# Generate random samples
n_samples = 500
x_samples = np.random.uniform(0, 10, n_samples)
y_samples = np.random.uniform(0, 10, n_samples)
labels = [is_inside(x, y, true_rect) for x, y in zip(x_samples, y_samples)]

# Create the animation
fig, ax = plt.subplots(figsize=(10, 8))

def init():
    """Initialize the plot."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('PAC Learning: Learning a Rectangle from Samples', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return []

def update(frame):
    """Update function for each animation frame."""
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Draw the true rectangle (semi-transparent)
    true_width = true_rect['x_max'] - true_rect['x_min']
    true_height = true_rect['y_max'] - true_rect['y_min']
    true_patch = Rectangle((true_rect['x_min'], true_rect['y_min']), 
                           true_width, true_height,
                           linewidth=2, edgecolor='green', 
                           facecolor='green', alpha=0.2, 
                           label='True Rectangle (unknown)')
    ax.add_patch(true_patch)
    
    # Get samples up to current frame
    n_current = min(frame + 1, n_samples)
    x_current = x_samples[:n_current]
    y_current = y_samples[:n_current]
    labels_current = labels[:n_current]
    
    # Separate positive and negative examples
    x_pos = [x for x, l in zip(x_current, labels_current) if l]
    y_pos = [y for y, l in zip(y_current, labels_current) if l]
    x_neg = [x for x, l in zip(x_current, labels_current) if not l]
    y_neg = [y for y, l in zip(y_current, labels_current) if not l]
    
    # Plot samples
    if x_pos:
        ax.scatter(x_pos, y_pos, c='blue', s=50, marker='o', 
                  label=f'Positive samples ({len(x_pos)})', zorder=5, alpha=0.6)
    if x_neg:
        ax.scatter(x_neg, y_neg, c='red', s=50, marker='x', 
                  label=f'Negative samples ({len(x_neg)})', zorder=5, alpha=0.6)
    
    # Compute and draw learned rectangle
    learned_rect = compute_learned_rect(x_pos, y_pos)
    if learned_rect:
        learned_width = learned_rect['x_max'] - learned_rect['x_min']
        learned_height = learned_rect['y_max'] - learned_rect['y_min']
        learned_patch = Rectangle((learned_rect['x_min'], learned_rect['y_min']), 
                                 learned_width, learned_height,
                                 linewidth=3, edgecolor='blue', 
                                 facecolor='blue', alpha=0.15,
                                 linestyle='--',
                                 label='Learned Rectangle')
        ax.add_patch(learned_patch)
        
        # Compute error
        error = compute_error(learned_rect, true_rect)
        error_text = f'Error: {error:.3f}'
    else:
        error_text = 'Error: N/A (no positive samples yet)'
    
    # Title with current state
    ax.set_title(f'PAC Learning: Sample {n_current}/{n_samples} | {error_text}', 
                fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=10)
    
    return []

# Create animation
print("Creating animation... This may take a minute.")
anim = FuncAnimation(fig, update, frames=n_samples, 
                    init_func=init, blit=True, 
                    interval=50, repeat=True)

# Save as GIF
writer = PillowWriter(fps=20)
print("Saving animation to 'rectangle_learning.gif'...")
anim.save('rectangle_learning.gif', writer=writer)
print("Animation saved successfully!")

# Also save the final frame as a static image
update(n_samples - 1)
plt.savefig('rectangle_learning_final.png', dpi=150, bbox_inches='tight')
print("Final frame saved as 'rectangle_learning_final.png'")

plt.close()
