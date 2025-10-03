import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from IPython.display import HTML

# Set random seed for reproducibility
np.random.seed(42)

# Define the true (unknown) circle
true_circle = {
    'center_x': 5.0,
    'center_y': 5.0,
    'radius': 3.0
}

# Function to check if a point is inside the true circle
def is_inside_circle(x, y, circle):
    distance = np.sqrt((x - circle['center_x'])**2 + (y - circle['center_y'])**2)
    return distance <= circle['radius']

# Generate random samples (only 50 for higher error)
n_samples = 50
x_samples = np.random.uniform(0, 10, n_samples)
y_samples = np.random.uniform(0, 10, n_samples)
labels = [is_inside_circle(x, y, true_circle) for x, y in zip(x_samples, y_samples)]

# Function to compute learned circle from positive examples
def compute_learned_circle(x_pos, y_pos):
    if len(x_pos) == 0:
        return None
    
    # Center: centroid of positive examples
    center_x = np.mean(x_pos)
    center_y = np.mean(y_pos)
    
    # Radius: maximum distance from center to any positive example
    distances = [np.sqrt((x - center_x)**2 + (y - center_y)**2) 
                 for x, y in zip(x_pos, y_pos)]
    radius = max(distances) if distances else 0
    
    return {
        'center_x': center_x,
        'center_y': center_y,
        'radius': radius
    }

# Function to compute error (area of symmetric difference)
def compute_error(learned_circle, true_circle):
    if learned_circle is None:
        return 1.0
    
    # Area of circles
    true_area = np.pi * true_circle['radius']**2
    learned_area = np.pi * learned_circle['radius']**2
    
    # Distance between centers
    center_dist = np.sqrt((true_circle['center_x'] - learned_circle['center_x'])**2 + 
                         (true_circle['center_y'] - learned_circle['center_y'])**2)
    
    # Calculate intersection area using geometric formula
    r1 = true_circle['radius']
    r2 = learned_circle['radius']
    d = center_dist
    
    if d >= r1 + r2:
        # No intersection
        intersection = 0
    elif d <= abs(r1 - r2):
        # One circle inside the other
        intersection = np.pi * min(r1, r2)**2
    else:
        # Partial intersection
        part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        intersection = part1 + part2 - part3
    
    # Symmetric difference normalized by true area
    error = (true_area + learned_area - 2 * intersection) / true_area
    return error

# Create the animation
fig, ax = plt.subplots(figsize=(10, 8))

def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('PAC Learning: Learning a Circle from Samples', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    return []

def update(frame):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Draw the true circle (semi-transparent)
    true_patch = Circle((true_circle['center_x'], true_circle['center_y']), 
                        true_circle['radius'],
                        linewidth=2, edgecolor='green', 
                        facecolor='green', alpha=0.2, 
                        label='True Circle (unknown)')
    ax.add_patch(true_patch)
    
    # Mark true center
    ax.plot(true_circle['center_x'], true_circle['center_y'], 
            'g+', markersize=15, markeredgewidth=2, label='True Center')
    
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
    
    # Compute and draw learned circle
    learned_circle = compute_learned_circle(x_pos, y_pos)
    if learned_circle:
        learned_patch = Circle((learned_circle['center_x'], learned_circle['center_y']), 
                              learned_circle['radius'],
                              linewidth=3, edgecolor='blue', 
                              facecolor='blue', alpha=0.15,
                              linestyle='--',
                              label='Learned Circle')
        ax.add_patch(learned_patch)
        
        # Mark learned center
        ax.plot(learned_circle['center_x'], learned_circle['center_y'], 
                'b+', markersize=15, markeredgewidth=2, label='Learned Center')
        
        # Compute error
        error = compute_error(learned_circle, true_circle)
        error_text = f'Error: {error:.3f}'
        
        # Additional info
        center_error = np.sqrt((learned_circle['center_x'] - true_circle['center_x'])**2 + 
                              (learned_circle['center_y'] - true_circle['center_y'])**2)
        radius_error = abs(learned_circle['radius'] - true_circle['radius'])
        
        info_text = f'Center offset: {center_error:.2f} | Radius error: {radius_error:.2f}'
    else:
        error_text = 'Error: N/A (no positive samples yet)'
        info_text = ''
    
    # Title with current state
    title = f'PAC Learning Circle: Sample {n_current}/{n_samples} | {error_text}'
    if info_text:
        title += f'\n{info_text}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=9)
    
    return []

# Create animation
anim = FuncAnimation(fig, update, frames=n_samples, 
                    init_func=init, blit=True, 
                    interval=100, repeat=True)

# Save as GIF (for GitHub repo)
from matplotlib.animation import PillowWriter

print("Creating animation... This may take a moment.")
writer = PillowWriter(fps=10)
print("Saving animation to 'circle_learning.gif'...")
anim.save('circle_learning.gif', writer=writer)
print("Animation saved successfully!")

# Also save the final frame as a static image
update(n_samples - 1)
plt.savefig('circle_learning_final.png', dpi=150, bbox_inches='tight')
print("Final frame saved as 'circle_learning_final.png'")

# Display the animation in notebook
plt.tight_layout()
HTML(anim.to_jshtml())
