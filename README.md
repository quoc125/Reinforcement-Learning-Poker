Repository worked on by Saeed Rahaman, Quoc Le, and Eric Watson

Files in Repository:

dqn.py 
   - stores the DQN agents for Model 1 and Model 2 and Model 3

network.py
   - stores the PyTorch network architectures for Model 1 and Model 2 and Model 3

players.py
   - file contains the player objects for call player, fold player, heuristic player, and random player 
   - bots used for DQN agents in dqn.py to train against

test_all_models.py
   - used to test all 3 models against each other after training 

test_model_1.py
   - tests DQN Model 1 against the four bots from players.py

test_model_2.py
   - tests DQN Model 2 against the four bots from players.py

test_model_3.py
   - tests DQN Model 3 against the four bots from players.py

train_model_1.py
   - trains DQN Model 1 for 5000 episodes against the four bots from players.py

train_model_2.py
   - trains DQN Model 2 for 5000 episodes against the four bots from players.py

train_model_3.py
   - trains DQN Model 3 for 5000 episodes against the four bots from players.py



To test each individual model run test_model_1, test_model_2 and test_model_3

To train each individual model run train_model_1, train_model_2, and train_model_3

To test all of the models run test_all_models
