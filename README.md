# Dragonfly Missile Command: Neuromorphic Controllers for Interception and Flight

[nc.pdf](Read as PDF)

## Authors

**M. Elijah Wangeman**  
BS Computer Science ([maw4678@rit.edu](mailto:maw4678@rit.edu))  

**Diana Velychko**  
MS Artificial Intelligence ([dv6943@rit.edu](mailto:dv6943@rit.edu))  

**Jake Streamer**  
BS Individualized Study ([jws8495@rit.edu](mailto:jws8495@rit.edu))  

**Nithyasri Narasimhan**  
MS Artificial Intelligence ([nn7985@rit.edu](mailto:nn7985@rit.edu))  

Rochester Institute of Technology  
Rochester, NY, USA

---

## Abstract
Dragonflies are uniquely effective, intelligent, and efficient at navigating a 3D environment to intercept a target. They reliably fly on an interception course unique across all observed pursuit predators. The dragonfly achieves this task with a brain of only 1 million neurons, seemingly computing trajectory prediction on as few as tens of thousands, through just 3 neural layers. The AKD1000 is a neuromorphic processor that can implement Spiking Neural Networks (SNN) on hardware. By modeling a dragonfly pursuit and training a Convolutional Neural Network (CNN) model to control the dragonfly purely from visual stimulus based on research from Dr. Chance, then converting that model to an Akida-compatible SNN, we evidence the potential of edge neuromorphic devices for spatial navigation and targeting.

---

## Introduction

Recent literature has become unusually interested in the mind of the common dragonfly. The dragonfly is a carnivorous insect that hunts and eats smaller insects. Zoological studies indicate that dragonflies catch approximately 95% of the prey they seek in the air - a remarkable success rate. This success rate is enabled by their speed and maneuverability, but also their impressive ability to chase prey on an intercepting trajectory, accurately predicting prey trajectory. Notably, they react to alterations in their prey's trajectory within 30-50 ms, demonstrating that a maximum of 3 layers of neurons is required for the dragonfly to make motor decisions while predicting the path of their prey.

Anti-ballistic missile systems have long relied on complex, computationally expensive, mathematics to intercept flying projectiles. These missile systems, while advanced, require enormous resources to fire and then to apply advanced position correction algorithms on scarce integrated systems hardware.

Drone interception is a more appropriate analogy to dragonfly interception. Unmanned Aerial Vehicles (UAVs) are rapidly advancing for a wide variety of applications. There is excellent potential for new, fast, power-efficient UAVs utilizing bioinspired, neuromorphic hardware systems.

Applications extend beyond these examples; modern camera drones rely heavily on machine vision for tracking subjects, a computationally expensive effort that is consequential for battery-powered, range-limited consumer drones.

---

## Methodology

Drones have limitations in weight, storage, and energy as these resources are sparse. Neuromorphic chip architecture permits reduced energy consumption and allows information storage using the time dimension. In our experiment, we use the AKD1000 chip, which is a neuromorphic processing chip developed by BrainChip Holdings. Unlike traditional GPUs, which rely on dense matrix operations, the AKD1000 processes sparse, event-driven data, making it much more efficient for real-world workloads. This approach not only reduces computational overhead but also aligns closely with the operation of the human brain, offering a pathway to more biologically inspired AI. The main reason to use the neuromorphic chip is its energy efficiency, which is crucial in edge devices such as UAVs.

Dragonfly brains have some unique characteristics that make models of them more straightforward than might initially be expected. Distinct neural pathways exist directly between the eyes and wings of a dragonfly, seemingly bypassing it's central nervous system entirely. This is theorized to be responsible for their fast reaction times while chasing, and points towards a relatively simple but very fast neural network we have to simulate: only considering that one pathway, ignoring the rest of the dragonfly nervous system, but still having an intelligent navigation model.

Dragonflies have some advantages, particularly in visual processing speed (200 hz) and a near 160$^\circ$ FOV (Which, in the interest of reasonable precision in tracking targets within a FOV, we are not replicating in our models). Research indicates, though, this comes at the cost of visual detail and depth perception. The lack of depth perception makes their trajectory predicting powers even more impressive, but also points to it not being a computationally complex 3D space trajectory solver, but instead some more novel (and, importantly, efficient) evolved approach of guaranteeing a collision course. 

### Neural Network

```python
X = np.load('data/X.npy')
y = np.load('data/y.npy')

model = Sequential([
    Input(shape=(FOV_SIZE, FOV_SIZE, 1)),
    Flatten(),
    Dense(441, activation='relu'),
    Dense(194481, activation='relu'),
    Dense(441, activation='relu'),
    Dense(2, activation='linear'),  # Output layer for pitch and yaw adjustments
    Reshape((2,))
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])

checkpoint = ModelCheckpoint('models/algo_trained.keras', monitor='val_loss', save_best_only=True, mode='min')

model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

model.save('models/algo_trained_final.keras')
```

![Training Epochs](https://github.com/elijah-code/scenario_out_3d/training-epochs.png)
*Figure 1: `algo_model` training that would have taken upwards of 3 hours per epoch on Google Colab.*

The proposed neural network mimics a dragonfly's prey interception system with a unique approach to tracking and capturing prey. It consists of four dense layers presented in the code snippet above. The first layer consists of 441 neurons representing the dragonfly's visual field, arranged in a 21 by 21 square array. This layer captures the prey's location within the dragonfly's field of view. In the second layer, we used $21^4$ (194481) neurons for the creation of the movement directions. A third set of 441 neurons hypothetically helps determine which neurons should be aligned with the prey's image. The final layer consists of 2 neurons that output yaw and rotation.

The network is first setup as an artificial neural network and then converted into the spiking neural network using Keras conversion function. 

### Keras Model to Akida SNN

```python
model_keras = keras.models.load_model('models/algo_trained.keras')

qparams = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)
model_quantized = quantize(model_keras, qparams=qparams)
model_quantized.summary()

model_akida = convert(model_quantized)
model_akida.summary()

model_akida.save('models/algo_trained_akida.fbz')
```

---

## Experimental Design

The algorithms and a neural network are tested in the 3D space within a simulated scenario.

### Simulated Environment

```python
class Scenario:
    def __init__(self, prey_trajectory, initial_dragonfly_pos, initial_dragonfly_heading, brain=brain_classic_direct):
        self.time = 0
        self.prey_trajectory = prey_trajectory
        self.brain = brain

        # Dragonfly initial state
        self.dragonfly_pos = initial_dragonfly_pos
        self.dragonfly_heading = initial_dragonfly_heading

        # Initialize dragonfly trajectory with starting position
        self.dragonfly_trajectory = np.array([initial_dragonfly_pos])
```

We generate linear and parabolical prey trajectories for testing the algorithm in 3D space. From a predefined initial point, we generate each step of the dragonfly using our algorithm. In the first state, we rotate the dragonfly such that its field of view captures the prey position. The main idea is to place the prey approximately in the center of its visual field. The field of view is set up to be a 2D space with a width and height of 21 pixels.

![Dragonfly Perspective](https://github.com/elijah-code/scenario_out_3d/fov.png)
*Figure 2: The dragonfly perspective computed from dragonfly and prey location and dragonfly heading.*

---

## Analysis

| Approach      | Brain                       | Model                  | Decision Time (ms) | Success Rate |
|---------------|-----------------------------|------------------------|--------------------|--------------|
| Akida         | `brain_akida()`            | `algo_trained_quantized` | 13.42             | 0.82         |
| Keras         | `brain_keras()`            | `algo_trained`         | 59.42             | 0.81         |
| Algorithm     | `brain_classic_direct()`   | N/A                    | 0.003344          | 0.72         |
| Keras         | `brain_keras()`            | `algo_trained_final`   | 70.52             | 0.61         |
| Algorithm     | `brain_offset_prev()`      | N/A                    | 0.0066            | 0.46         |

---

This table provides an overview of a subset of our approaches and their outcomes. One somewhat unexpected result was the capability of the simple discrete proportional control algorithm in capturing the prey; for very low computational cost, it competes within 0.2 of the success rate of both our and Dr. Chance's dragonflies, and within 0.25 of the success rate of actual dragonflies (Don't take this comparison seriously, neither model is equivalent to actual dragonfly hunting conditions). Training on pruned datasets enabled some measure of improvement, but stayed within 0.08 of the classical results. It's worth mentioning that the increase from the floating point algo_trained keras model on M1 Max to it's quantized sibling on the AKD1000 is just chance- the results skew 3-4% across tests. These tests were ran with the `count_win_fail_states()` function, visible within the Scenario class on the Github.

The results of the on-chip neuromorphic inferencing are ideal and further evidence the value of this project. 13.42 ms is a decision time that makes real-time navigation on device not just feasible, but currently implementable. It does this while losing no performance capabilities quantized to 8 bits and comfortably fitting on the AKD1000. These are very compelling results considering how little we have optimized the models for that silicon, coming from a training infrastructure based on Dr. Chances research and for the most part developed without special considerations for the AKD1000. 

Comparing `algo_trained_final` and `algo_trained` provides a very clear example of overfitting. The `rand_final` model puts into perspective how intelligent and successful even relatively uncompetitive algorithms like our offset fov approach are, which was an attempt at replicating Dr. Chances second layer of neurons that enabled offset targeting, which allows predictive instead of pursuit interceptions. Unfortunately, our non-neural approach significantly outperforms the direct pursuit algorithm 25% of the time (This isn't a coincidence, just a result of the number of cardinal directions) and significantly underperforms 75% of the time.