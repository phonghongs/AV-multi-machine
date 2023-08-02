from Script.Vehicles.Bicycle_model import BicycleModel

x = [0, 1, 3, 5]
y = [0, 0.25, 0.5, 0.75]

model = BicycleModel()

model.GetOptimizeSteering(6, x, y)