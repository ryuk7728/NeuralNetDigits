import pygame
import numpy as np
import sys
import os

# Initialize Pygame
pygame.init()

# Set up display
width, height = 560, 560  # 20x scale for visibility
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("28x28 Pixel Drawing Grid")

# Create a 28x28 numpy array to store grayscale values
grid = np.zeros((28, 28), dtype=np.uint8)

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Function to load existing data from data.py
def load_existing_data():
    if os.path.exists('data.py'):
        with open('data.py', 'r') as f:
            exec(f.read(), globals())
        if 'x_train' in globals() and 'y_train' in globals():
            print("x_train",x_train.shape,y_train.shape)
            return x_train, y_train
    return np.empty((0, 784)), np.empty((0, 10))

# Function to save data to data.py
def save_to_data_py(x_train, y_train):
    with open('data.py', 'w') as f:
        f.write("import numpy as np\n\n")
        f.write(f"x_train = np.array({x_train.tolist()})\n")
        f.write(f"y_train = np.array({y_train.tolist()})\n")
        f.write("\n# Save the training data\n")
        f.write("np.save('x_train.npy', x_train)\n")
        f.write("np.save('y_train.npy', y_train)\n")

# Load existing data
x_train, y_train = load_existing_data()
print("data: ",x_train,y_train)

# Main loop
drawing = False
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            x, y = event.pos
            grid_x, grid_y = x // 20, y // 20
            if 0 <= grid_x < 28 and 0 <= grid_y < 28:
                grid[grid_y, grid_x] = 255  # Set to white
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                # Prompt for the digit
                digit = input("What digit did you draw (0-9)? ")
                try:
                    digit = int(digit)
                    if 0 <= digit <= 9:
                        # Create one-hot encoded label
                        label = np.zeros(10)
                        label[digit] = 1
                        
                        # Add to training data
                        x_train = np.vstack([x_train, grid.reshape(1, 784)])
                        y_train = np.vstack([y_train, label.reshape(1, 10)])
                        
                        print(f"Digit {digit} added to training data.")
                        grid.fill(0)  # Clear the grid
                    else:
                        print("Please enter a digit between 0 and 9.")
                except ValueError:
                    print("Invalid input. Please enter a digit between 0 and 9.")
            elif event.key == pygame.K_c:
                grid.fill(0)  # Clear the grid
            elif event.key == pygame.K_q:
                # Save and quit
                if x_train.size > 0 and y_train.size > 0:
                    save_to_data_py(x_train, y_train)
                    print("Training data saved to data.py")
                    print(f"Total samples: {x_train.shape[0]}")
                else:
                    print("No training data to save.")
                running = False

    # Draw the grid
    screen.fill(BLACK)
    for y in range(28):
        for x in range(28):
            color = (grid[y, x], grid[y, x], grid[y, x])
            pygame.draw.rect(screen, color, (x*20, y*20, 20, 20))
    
    # Draw grid lines
    for i in range(29):
        pygame.draw.line(screen, (100, 100, 100), (i*20, 0), (i*20, height))
        pygame.draw.line(screen, (100, 100, 100), (0, i*20), (width, i*20))
    
    pygame.display.flip()

pygame.quit()
sys.exit()