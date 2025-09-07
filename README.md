# Overview
A TensorFlow implementation of a Temporal Convolutional Network (TCN) to classify individuals based on their gait patterns in the Har-Lab Dataset.  
**Not activity-specific:** gait patterns are tied to individuals across activities.
  
**Performance Evaluation**  
(10 ms per timestep)
* `timesteps=100`: 92% test accuracy
* `timesteps=200`: 96% test accuracy
* `timesteps=300`: 98% test accuracy
  
This is actually really interesting, as gait cycles are typically 1-1.5 seconds, potentially explaining the significant accuracy increase between `timesteps=100` and `timesteps=200` (4%), as opposed to the mild increase between `timesteps=200` and `timesteps=300` (2%).
  
**Credibility**: since data was collected in contiguous sequences per activity/person, it's difficult to define a set for testing. As a reasonable approach, the last 20% of each activity sequence per person (walk_mixed, walk_sidewalk, walk_treadmill), is reserved in a contiguous block to evaluate the model. This approach shows reasonable proof of generalizing and learning as opposed to that of randomly splitting sequences into train and test, which reaches an unlikely 99% test accuracy after 3 epochs with `timesteps=100`.  

**Required packages:** pandas, numpy, tensorflow, keras