from spiral_datapoint import SpiralData

# Create an instance of SpiralData
spiral_data = SpiralData(num_points=250, noise=0.2, revolutions=4)
spiral_data.generate_data()
spiral_data.plot_data()

 # Get the coordinates
(x1, y1), (x2, y2) = spiral_data.get_coordinates()