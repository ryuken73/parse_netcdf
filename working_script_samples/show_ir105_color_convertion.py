import matplotlib.pyplot as plt
import numpy as np

# get_color_from_temperature 함수
def get_color_from_temperature(temp):
    # Define key color points and their RGB values based on the gradient
    colors = {
        20: [0, 0, 0],       # Black
        -20: [255, 255, 255], # White
        -21: [135, 206, 235], # Sky blue (sharp transition from white at -20)
        -30: [0, 0, 255],    # Blue
        -40: [0, 255, 0],    # Green
        -45: [144, 238, 144], # Light green
        -50: [255, 255, 0],   # Yellow
        -60: [255, 0, 0],     # Red
        -70: [0, 0, 0],       # Black
        -80: [255, 255, 255], # White (sharp transition from black at -70)
        -81: [128, 128, 128], # Gray (sharp transition from white at -80)
        -90: [128, 0, 128]    # Purple
    }
    
    # Clamp temperature to valid range (20 to -90)
    temp = max(-90, min(20, temp))
    
    # Find the two closest key points for interpolation (considering sharp transitions)
    keys = sorted(colors.keys(), reverse=True)  # Sort in descending order (20 to -90)
    for i in range(len(keys) - 1):
        if temp <= keys[i] and temp > keys[i + 1]:
            start_temp, end_temp = keys[i], keys[i + 1]
            start_color, end_color = colors[start_temp], colors[end_temp]
            break
    else:
        if temp <= -81:
            start_temp, end_temp = -81, -90
            start_color, end_color = colors[-81], colors[-90]
        elif temp >= 20:
            return [0, 0, 0, 255]  # Black for temp >= 20
        else:
            start_temp, end_temp = keys[0], keys[1]
            start_color, end_color = colors[start_temp], colors[end_temp]
    
    # Linear interpolation based on temperature position
    if start_temp == end_temp or (start_temp in [-20, -80] and temp == start_temp + 1):
        return end_color + [255]  # Sharp transition at -20 and -80
    else:
        ratio = (start_temp - temp) / (start_temp - end_temp)  # Adjust for descending order
        r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
        g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
        b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
        return [r, g, b, 255]

# Create a dense temperature array for smooth gradient
temperatures = np.linspace(20, -90, 1000)  # High resolution for smooth transition
colors = [get_color_from_temperature(t) for t in temperatures]

# Extract RGB values (normalize to 0-1 range for matplotlib)
rgb_colors = np.array([[c[0]/255, c[1]/255, c[2]/255] for c in colors])

# Create a plot with a smooth gradient
fig, ax = plt.subplots(figsize=(10, 2))

# Use imshow for a continuous color bar
gradient = np.tile(rgb_colors, (10, 1, 1))  # Repeat to create a wider bar
ax.imshow(gradient, aspect='auto', extent=[20, -90, 0, 1], interpolation='bicubic')
ax.set_xlim(20, -90)
ax.set_xticks(np.arange(20, -91, -10))
ax.set_xlabel('Temperature (°C)')
ax.set_yticks([])  # Remove y-axis

# Add a title
plt.title('Temperature Color Gradient')

# Display the plot
plt.show()