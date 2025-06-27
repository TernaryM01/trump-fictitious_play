"Trump" is a 4-player trick-taking card game which involves careful planning, bluffing, etc. This project builds a player AI to play the game skillfully using a self-play deep learning algorithm adapted from [the CFVFP paper](https://paperswithcode.com/paper/pure-monte-carlo-counterfactual-regret).

Features:
1. Efficient implementation with C++ environment, JAX neural network training, and parallel game traversals. The dataloader is manually handcrafted, with buffer and neural network weights files automatically saved, loaded, pruned, and cleaned up. Buffer data to be fed to the neural networks are collected from files and then concatenated and batched from on the GPU to maximize speed; data augmentation (see next point) happens when preparing each batch.
2. Flexible & generic game solver class that can work with any game, different neural networks for different kinds of game states (e.g., in the setup for the Trump game, the trick-taking phase is handled by a different neural network architecture than the bidding phase), a "revelation transformer" that reveals hidden information by transforming a player's infostate tensor by taking information from other players' infostate tensors (empirically found to be beneficial for speeding up training), and a data augmentor (in the Trump game setup, it permutes card suits as long as that doesn't change the highest bidder).
3. A powerful feature engineering for the Trump game is the current-trick-taking (un)certainty. It is implemented using a Z3 logic program preceded by faultless speedy heuristics (that are used to handle easy cases before falling back to the logic program for hard cases). The feature is implemented as an infostate transformer, and so it works seamlessly with the revelation transformer and takes full advantage of all the information revealed by it.

To Do:
1. The Z3 logic program has to be made much faster. It's currently very slow.
2. Use state "determinization" with sound Bayesian reasoning to weight possible states. This enables planning and 'reading' other players' cards based on their behaviors. Implementing this will involve providing the "ResampleFromInfostate" function in the Open Spiel environment and weight calculation using trained policies.
3. Make the "How to Use" instruction (see below) much clearer and beginner-friendly.
4. Perhaps a better neural network architecture and better optimizers?

How to Use:
1. Build Open Spiel (clone the repository and "pip install" it) with the Trump environment (`trump.cc` and `trump.h`) included. Look up Open Spiel documentary to find out how to do this.
2. Install other packages: Numpy, JAX, Z3, etc.
3. Play with the Jupyter notebook provided.
