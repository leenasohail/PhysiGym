# Vec + SAC

## Acceleration Thanks to Async SAC ( HAVE TO WORK ON IT)
[Our current DRL](VectorizedDRL.png), it is great but it is not efficient. We are spending too much time between neural network updates and environment steps.

Two main papers:  
- [Asynchronous Parallel Reinforcement](https://link.springer.com/article/10.1007/s00366-024-02093-w?utm_source=chatgpt.com)  
- [Asynchronous A3C on GPU](https://www.researchgate.net/profile/Iuri-Frosio/publication/310610848_GA3C_GPU-based_A3C_for_Deep_Reinforcement_Learning/links/583c6c0b08ae502a85e3dbb9/GA3C-GPU-based-A3C-for-Deep-Reinforcement-Learning.pdf)

What these article do:
[AsynchronousParallelTraining_APT](AsynchronousParallelTraining_APT.png)
As we know, the simulator should run without any interruptions and feed the replay buffer continuously, without waiting. This should be done asynchronously. In the end, the different PhysiCell simulations should run non-stop.


# My Plan
- Putting test env for instance the last env
- Adding a distribution of initial conditions
- Make the code more similar to Async SAC, inspiration from the two articles above.
- Add PPO.
